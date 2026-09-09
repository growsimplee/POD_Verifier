# POD_Verifier

AWS Lambda pipeline for POD quality scoring: **Sarathy POD verification request → single container Lambda (per-trip image download → EfficientNet inference) → PostgreSQL `pod_scores`**.

Scoring is event-driven per trip (the daily cron sweep is kept but disabled, for backfills).

See **[aws/README.md](aws/README.md)** for architecture and deployment (CloudFormation + Docker/ECR, **no SAM**).

## Deploying

Merge to `stag-main` for staging (automatic) or `prod-main` for production (one
approval). CircleCI creates everything the service needs — ECR repository,
CloudFormation stack, Lambda, schedules, alarms and the database schema — from
nothing. See **[aws/DEPLOYMENT.md](aws/DEPLOYMENT.md)**.

## Repository layout

- `aws/` — Lambda image (`lambda_scorer/`), **CloudFormation `infra/stack.yaml`**, **`provision-stack.sh`** / **`deploy.sh`**, PostgreSQL schema.

**Metabase**, **PostgreSQL**, **batch sizes**, and **score thresholds** are **Lambda environment variables** set by **`provision-stack.sh`** / `stack.yaml`. Do not commit real secrets (`config.env.example` — placeholders only).

## Trained model in Git

**`aws/lambda_scorer/model/best.pt`** is the canonical checkpoint (see repo `.gitignore` exception).

```bash
cp /path/to/your/trained/best.pt aws/lambda_scorer/model/best.pt
git add aws/lambda_scorer/model/best.pt && git commit -m "Update trained checkpoint" && git push
```

Use **Git LFS** or private artifacts if the file is very large.
