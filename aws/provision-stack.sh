#!/usr/bin/env bash
# Create or update the POD pipeline stack (plain CloudFormation — no SAM).
#
# Prereqs: AWS CLI v2, credentials, image already in ECR (see deploy.sh).
#
# Required env (example):
#   export AWS_REGION=ap-south-1
#   export STACK_NAME=pod-scoring-prod
#   export STAGE=prod
#   export VPC_ID=vpc-xxx
#   export SUBNET_IDS=subnet-a,subnet-b
#   # sarathy's internal API — this function holds no DB credentials and no SQL
#   export SARATHY_BASE_URL=http://sarathy.internal:8080
#   # optional: IAM principal (Sarathy) allowed to invoke the function
#   export INVOKER_PRINCIPAL_ARNS=arn:aws:iam::123456789012:role/sarathy-task-role
#   export SCORER_IMAGE_URI=123456789012.dkr.ecr.ap-south-1.amazonaws.com/pod-pipeline:latest
#
# Optional: SARATHY_TIMEOUT SARATHY_RETRIES SARATHY_PAGE_SIZE
#           TRIP_MAX_DOWNLOAD_WORKERS WARM_POOL_SIZE WARMUP_RATE
#           LOG_RETENTION_DAYS ALARM_EMAIL
#           BACKFILL_SCHEDULE BACKFILL_TIMEZONE BACKFILL_STATE
#           INVOKER_PRINCIPAL_ARNS RESERVED_CONCURRENCY FLAG_THRESHOLD
#           INFERENCE_BATCH_SIZE MAX_DOWNLOAD_WORKERS WINDOW_SIZE IMAGENET_NORMALIZE
#           TMP_EPHEMERAL_MB
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${AWS_DIR}/infra/stack.yaml"

AWS_REGION="${AWS_REGION:-us-east-2}"
STACK_NAME="${STACK_NAME:-pod-scoring-stg}"
STAGE="${STAGE:-stg}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
SARATHY_BASE_URL="${SARATHY_BASE_URL:-}"
SARATHY_TIMEOUT="${SARATHY_TIMEOUT:-15}"
SARATHY_RETRIES="${SARATHY_RETRIES:-3}"
SARATHY_PAGE_SIZE="${SARATHY_PAGE_SIZE:-500}"
TRIP_MAX_DOWNLOAD_WORKERS="${TRIP_MAX_DOWNLOAD_WORKERS:-8}"
WARM_POOL_SIZE="${WARM_POOL_SIZE:-3}"
WARMUP_RATE="${WARMUP_RATE:-rate(5 minutes)}"
BACKFILL_SCHEDULE="${BACKFILL_SCHEDULE:-cron(15 0 * * ? *)}"
BACKFILL_TIMEZONE="${BACKFILL_TIMEZONE:-Asia/Kolkata}"
BACKFILL_STATE="${BACKFILL_STATE:-DISABLED}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"
ALARM_EMAIL="${ALARM_EMAIL:-}"
INVOKER_PRINCIPAL_ARNS="${INVOKER_PRINCIPAL_ARNS:-}"
RESERVED_CONCURRENCY="${RESERVED_CONCURRENCY:-10}"
FLAG_THRESHOLD="${FLAG_THRESHOLD:-0.7}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
MAX_DOWNLOAD_WORKERS="${MAX_DOWNLOAD_WORKERS:-64}"
WINDOW_SIZE="${WINDOW_SIZE:-800}"
IMAGENET_NORMALIZE="${IMAGENET_NORMALIZE:-true}"
TMP_EPHEMERAL_MB="${TMP_EPHEMERAL_MB:-512}"   # in-memory design uses no /tmp
SCORER_IMAGE_URI="${SCORER_IMAGE_URI:-}"

if [[ -z "$VPC_ID" || -z "$SUBNET_IDS" ]]; then
  echo "Set VPC_ID and SUBNET_IDS (comma-separated private subnets)." >&2
  exit 1
fi
if [[ -z "$SARATHY_BASE_URL" ]]; then
  echo "Set SARATHY_BASE_URL (sarathy's internal API base, reachable from SUBNET_IDS)." >&2
  exit 1
fi
if [[ "$SARATHY_BASE_URL" == https://* ]] && [[ "$SARATHY_BASE_URL" != *.internal* ]]; then
  echo "NOTE: /internal/pod-scoring has no app-level auth; SARATHY_BASE_URL must not be publicly routable." >&2
fi
if (( WARM_POOL_SIZE >= RESERVED_CONCURRENCY )); then
  echo "WARM_POOL_SIZE ($WARM_POOL_SIZE) must be below RESERVED_CONCURRENCY ($RESERVED_CONCURRENCY)." >&2
  exit 1
fi
if [[ -z "$SCORER_IMAGE_URI" ]]; then
  echo "Set SCORER_IMAGE_URI (same ECR tag you pushed with aws/deploy.sh)." >&2
  exit 1
fi

OVERRIDES=(
  "Stage=${STAGE}"
  "SarathyBaseUrl=${SARATHY_BASE_URL}"
  "SarathyTimeout=${SARATHY_TIMEOUT}"
  "SarathyRetries=${SARATHY_RETRIES}"
  "SarathyPageSize=${SARATHY_PAGE_SIZE}"
  "TripMaxDownloadWorkers=${TRIP_MAX_DOWNLOAD_WORKERS}"
  "WarmPoolSize=${WARM_POOL_SIZE}"
  "WarmupRate=${WARMUP_RATE}"
  "BackfillSchedule=${BACKFILL_SCHEDULE}"
  "BackfillTimezone=${BACKFILL_TIMEZONE}"
  "BackfillState=${BACKFILL_STATE}"
  "LogRetentionDays=${LOG_RETENTION_DAYS}"
  "AlarmEmail=${ALARM_EMAIL}"
  "InvokerPrincipalArns=${INVOKER_PRINCIPAL_ARNS}"
  "ReservedConcurrency=${RESERVED_CONCURRENCY}"
  "FlagThreshold=${FLAG_THRESHOLD}"
  "InferenceBatchSize=${INFERENCE_BATCH_SIZE}"
  "MaxDownloadWorkers=${MAX_DOWNLOAD_WORKERS}"
  "WindowSize=${WINDOW_SIZE}"
  "ImagenetNormalize=${IMAGENET_NORMALIZE}"
  "VpcId=${VPC_ID}"
  "SubnetIds=${SUBNET_IDS}"
  "ScorerImageUri=${SCORER_IMAGE_URI}"
  "TmpEphemeralMB=${TMP_EPHEMERAL_MB}"
)

echo "==> cloudformation deploy stack=${STACK_NAME} region=${AWS_REGION}"
aws cloudformation deploy \
  --stack-name "${STACK_NAME}" \
  --template-file "${TEMPLATE}" \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_IAM \
  --region "${AWS_REGION}" \
  --parameter-overrides "${OVERRIDES[@]}"

echo "==> outputs"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs" \
  --output table

echo "Done."
