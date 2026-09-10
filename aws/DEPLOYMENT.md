# POD Verifier — Lambda Deployment (no SAM)

Deployment-ready module for the **normalization-fixed** single-invocation pipeline.
Infra is plain **CloudFormation** + Docker + AWS CLI — **no SAM, no CDK**.

## What ships

- **Container Lambda** (`lambda_scorer/`): `handler.py` (single-invocation, resilient,
  **ImageNet-normalized inference by default**), `src/model.py`, `model/best.pt`.
- **`Dockerfile`**: CPU-only torch 2.2.2 / torchvision 0.17.2 (pinned), `numpy<2`;
  build-time check that deps import and the checkpoint loads with 0 key mismatches.
- **Infra** (`infra/stack.yaml`): VPC Lambda (10 GB / 900 s), event-driven per-trip
  invocation from Sarathy, a warm-pool schedule, a **DISABLED** EventBridge daily
  trigger kept for backfills, async retries + SQS DLQ, IAM, an explicit log group
  with retention, and alarms (errors, throttles, DLQ depth, uncovered images) on
  an SNS topic.
- **Scripts**: `deploy.sh` (ECR repo + build + push), `provision-stack.sh`
  (`aws cloudformation deploy`), `teardown.sh` (destroy an environment).

## The normal path: merge and walk away

```
stag-main  →  build + test  →  release           (staging, us-east-2, automatic)
prod-main  →  build + test  →  approve → release (prod, ap-south-1)
any other  →  build + test only, no AWS writes
```

`release` is idempotent and safe against a completely empty account. It creates
the ECR repository if it is missing, pushes this commit's image, creates or
updates the CloudFormation stack pointed at that exact tag, applies the database
schema, and smoke-tests the result. There is no manual "create the repo" or "run
schema.sql" step.

The schema is applied by invoking the function with `{"migrate": true}` rather
than by connecting to Postgres from CI. The database sits in a private subnet
CircleCI cannot reach; the Lambda is already inside the VPC, so it is the one
thing that can. `schema.sql` is baked into the image at `/opt/schema/schema.sql`
(staged from `infra/schema.sql` by `deploy.sh` — edit `infra/schema.sql`, never
the staged copy).

### What CI still cannot invent

The platform it runs on. These come from the CircleCI context, and `preflight`
fails on the first job naming every one that is missing:

| Variable | What it is |
|---|---|
| `VPC_ID` | The VPC the function runs in |
| `SUBNET_IDS` | Comma-separated private subnets, with NAT egress and a route to the RDS |

The database needs nothing added: `DBHOST`, `DB_PASSWORD`, `DBPORT`,
`DB_USERNAME` and `SARATHY_DBNAME` are already in the org contexts and are
mapped onto `PG_*` automatically. `pod_scores` is created in the sarathy
database, beside the `kaptaan` table it reads.

The three queries need nothing added either — `SOURCE_QUERY`, `TRIP_QUERY` and
`RANGE_QUERY` all default to `kaptaan` in `infra/stack.yaml`. Override one in a
context only when an environment should score something different.

Credentials come from the org contexts (`Aws-stage`, `Aws-prod`), which name
them `ACCESS_KEY_ID` / `AWS_ACCESS_KEY`. Those keys were created for sarathy's
build-and-push; this pipeline also needs CloudFormation, IAM, Logs, SNS,
CloudWatch, SQS, EC2 and Scheduler. An `AccessDenied` in `release` means the
policy needs widening.

## IAM for the CI user

The org's CircleCI keys were provisioned for build-and-push. This pipeline also
creates a CloudFormation stack, so the deploying principal needs more.
[`infra/ci-policy.json`](infra/ci-policy.json) is that policy — scoped to
`pod-*` resources rather than granted account-wide, and reusable in both
accounts since every ARN wildcards the account and region.

```bash
aws iam create-policy \
  --policy-name PodVerifierDeploy \
  --policy-document file://infra/ci-policy.json

aws iam attach-user-policy \
  --user-name circleCi \
  --policy-arn arn:aws:iam::<account>:policy/PodVerifierDeploy
```

CloudFormation creates the stack's resources as the calling principal (there is
no service role on the stack), which is why the user needs Lambda, IAM, Logs,
SNS, SQS, CloudWatch, EC2 and Scheduler permissions and not just
`cloudformation:*`. The `iam:*` grants are restricted to the two roles the stack
owns — `pod-pipeline-fn-*` and `pod-pipeline-scheduler-*` — and `iam:PassRole`
additionally requires the target service to be Lambda or Scheduler, so these
keys cannot mint a role for anything else.

Longer term this is better done with a CloudFormation **service role**: the CI
user gets `cloudformation:*` on these stacks plus `iam:PassRole` on one role,
and that role holds the resource permissions. Better still, replace the
long-lived user keys with OIDC.

## Prerequisites for running the scripts by hand

- Docker (with buildx) running.
- AWS CLI v2 authenticated with ECR/Lambda/CFN/IAM permissions.
- `lambda_scorer/model/best.pt` present (baked into the image).
- Subnets with egress to the RDS endpoint and the POD image host (NAT or VPC endpoints).

## Doing it manually

The same three steps CI runs, in order:

```bash
cd aws && chmod +x deploy.sh provision-stack.sh teardown.sh

export AWS_REGION=ap-south-1 STAGE=prod STACK_NAME=pod-scoring-prod
export ECR_REPOSITORY=pod-pipeline IMAGE_TAG=$(git rev-parse --short=12 HEAD)
export VPC_ID=vpc-xxx SUBNET_IDS=subnet-a,subnet-b
export PG_HOST=<db>.rds.amazonaws.com PG_PASSWORD=***
export INVOKER_PRINCIPAL_ARNS=arn:aws:iam::<acct>:role/<sarathy-prod-role>

SKIP_LAMBDA_UPDATE=true ./deploy.sh          # creates the ECR repo, pushes

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export SCORER_IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
./provision-stack.sh                          # creates the stack + function

aws lambda invoke --function-name pod-pipeline-prod --region "$AWS_REGION" \
  --payload '{"migrate": true}' --cli-binary-format raw-in-base64-out /dev/stdout
```

## Tearing an environment down

`teardown.sh` reverses all of it, including **dropping `pod_scores` and
`pod_scores_flagged`** — every score computed so far. It drops the table first,
while the function that can reach the database still exists, then deletes the
stack, the ECR repository and any orphaned log group.

```bash
cd aws
AWS_REGION=us-east-2 STAGE=stg STACK_NAME=pod-scoring-stg ./teardown.sh
#   KEEP_DATABASE=true   leave pod_scores alone
#   KEEP_ECR=true        keep the repository and its images
```

It asks you to type the stack name before doing anything. The VPC, the RDS
instance, `kaptaan` and the CircleCI contexts are left alone —
they belong to the platform, not to this service.

## Routine updates

Build, push, and roll the Lambda to the new image digest:

```bash
cd aws && ./deploy.sh
```

`DRY_RUN=true ./deploy.sh` builds the image locally only (no AWS) — a good pre-push smoke test.

## Verify after deploy

```bash
# Manual run (same payload the scheduler sends)
aws lambda invoke --function-name pod-pipeline-prod \
  --payload '{}' --cli-binary-format raw-in-base64-out /dev/stdout

# Rows land in Postgres
psql ... -c "SELECT status, count(*) FROM pod_scores WHERE run_date=CURRENT_DATE GROUP BY status;"

# Coverage metrics (CloudWatch namespace 'PODPipeline'): ImagesScored/Failed/Uncovered
```

Expect the response body to report `status=complete` with `scored + failed == total`.
If it reports `continuing`, a bounded continuation was queued (large day) — it resumes automatically.

## Operating notes (from the gold-set evaluation)

- **ImageNet normalization is on by default** (`IMAGENET_NORMALIZE=true`) — this is the
  correctness fix; do not set it to `false` in production.
- **FLAG threshold**: the eval favours ~0.55–0.60 (precision-first for penalties) over the
  0.7 default. Tune `FLAG_THRESHOLD` and the `0.7` in the `pod_scores_flagged` view together.
- **Coverage**: download failures are recorded (`status='download_failed'`) and excluded from
  PASS/FLAG, so an AWB is never penalised on a failed fetch.

## Rollback

Re-point the function at the previous image tag:

```bash
aws lambda update-function-code --function-name pod-pipeline-prod \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/pod-pipeline:<previous-tag> --region "$AWS_REGION"
```

CloudFormation changes roll back via `aws cloudformation deploy` of the prior template or the console.
