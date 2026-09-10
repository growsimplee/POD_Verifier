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
#   # all three queries have defaults; override only to change what is scored:
#   export SOURCE_QUERY="SELECT awb, trip_id, pod FROM kaptaan WHERE tour_date = CURRENT_DATE"
#   # optional, has a sane default; MUST keep the %(trip_id)s placeholder:
#   export TRIP_QUERY="SELECT awb, trip_id, pod FROM kaptaan WHERE trip_id = %(trip_id)s"
#   # optional: IAM principal (Sarathy) allowed to invoke the function
#   export INVOKER_PRINCIPAL_ARNS=arn:aws:iam::123456789012:role/sarathy-task-role
#   export PG_HOST=db.xxx.rds.amazonaws.com
#   export PG_PASSWORD=secret
#   export SCORER_IMAGE_URI=123456789012.dkr.ecr.ap-south-1.amazonaws.com/pod-pipeline:latest
#
# Optional: TRIP_QUERY RANGE_QUERY ALLOW_ADHOC_QUERY RESCORE_LOOKBACK_DAYS
#           TRIP_MAX_DOWNLOAD_WORKERS WARM_POOL_SIZE WARMUP_RATE
#           LOG_RETENTION_DAYS ALARM_EMAIL
#           SOURCE_PG_HOST SOURCE_PG_DATABASE SOURCE_PG_USER SOURCE_PG_PASSWORD
#           INVOKER_PRINCIPAL_ARNS RESERVED_CONCURRENCY FLAG_THRESHOLD
#           INFERENCE_BATCH_SIZE MAX_DOWNLOAD_WORKERS WINDOW_SIZE IMAGENET_NORMALIZE
#           PG_PORT PG_DATABASE PG_USER TMP_EPHEMERAL_MB
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${AWS_DIR}/infra/stack.yaml"

AWS_REGION="${AWS_REGION:-us-east-2}"
STACK_NAME="${STACK_NAME:-pod-scoring-stg}"
STAGE="${STAGE:-stg}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
SOURCE_QUERY="${SOURCE_QUERY:-SELECT awb, trip_id, pod FROM kaptaan WHERE tour_date = CURRENT_DATE}"
TRIP_QUERY="${TRIP_QUERY:-SELECT awb, trip_id, pod FROM kaptaan WHERE trip_id = %(trip_id)s}"
RANGE_QUERY="${RANGE_QUERY:-SELECT awb, trip_id, pod FROM kaptaan WHERE tour_date BETWEEN %(start_date)s AND %(end_date)s}"
ALLOW_ADHOC_QUERY="${ALLOW_ADHOC_QUERY:-true}"
RESCORE_LOOKBACK_DAYS="${RESCORE_LOOKBACK_DAYS:-30}"
TRIP_MAX_DOWNLOAD_WORKERS="${TRIP_MAX_DOWNLOAD_WORKERS:-8}"
WARM_POOL_SIZE="${WARM_POOL_SIZE:-3}"
WARMUP_RATE="${WARMUP_RATE:-rate(5 minutes)}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"
ALARM_EMAIL="${ALARM_EMAIL:-}"
INVOKER_PRINCIPAL_ARNS="${INVOKER_PRINCIPAL_ARNS:-}"
RESERVED_CONCURRENCY="${RESERVED_CONCURRENCY:-10}"
FLAG_THRESHOLD="${FLAG_THRESHOLD:-0.7}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
MAX_DOWNLOAD_WORKERS="${MAX_DOWNLOAD_WORKERS:-64}"
WINDOW_SIZE="${WINDOW_SIZE:-800}"
IMAGENET_NORMALIZE="${IMAGENET_NORMALIZE:-true}"
PG_HOST="${PG_HOST:-}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_PORT="${PG_PORT:-5432}"
PG_DATABASE="${PG_DATABASE:-pod_classifier}"
PG_USER="${PG_USER:-postgres}"
SOURCE_PG_HOST="${SOURCE_PG_HOST:-}"
SOURCE_PG_DATABASE="${SOURCE_PG_DATABASE:-}"
SOURCE_PG_USER="${SOURCE_PG_USER:-}"
SOURCE_PG_PASSWORD="${SOURCE_PG_PASSWORD:-}"
TMP_EPHEMERAL_MB="${TMP_EPHEMERAL_MB:-512}"   # in-memory design uses no /tmp
SCORER_IMAGE_URI="${SCORER_IMAGE_URI:-}"

if [[ -z "$VPC_ID" || -z "$SUBNET_IDS" ]]; then
  echo "Set VPC_ID and SUBNET_IDS (comma-separated private subnets)." >&2
  exit 1
fi
if [[ "$TRIP_QUERY" != *"%(trip_id)s"* ]]; then
  echo "TRIP_QUERY must contain the named placeholder %(trip_id)s." >&2
  exit 1
fi
if [[ "$RANGE_QUERY" != *"%(start_date)s"* || "$RANGE_QUERY" != *"%(end_date)s"* ]]; then
  echo "RANGE_QUERY must contain the placeholders %(start_date)s and %(end_date)s." >&2
  exit 1
fi
if (( WARM_POOL_SIZE >= RESERVED_CONCURRENCY )); then
  echo "WARM_POOL_SIZE ($WARM_POOL_SIZE) must be below RESERVED_CONCURRENCY ($RESERVED_CONCURRENCY)." >&2
  exit 1
fi
if [[ -z "$PG_HOST" || -z "$PG_PASSWORD" ]]; then
  echo "Set PG_HOST and PG_PASSWORD." >&2
  exit 1
fi
if [[ -z "$SCORER_IMAGE_URI" ]]; then
  echo "Set SCORER_IMAGE_URI (same ECR tag you pushed with aws/deploy.sh)." >&2
  exit 1
fi

OVERRIDES=(
  "Stage=${STAGE}"
  "SourceQuery=${SOURCE_QUERY}"
  "TripQuery=${TRIP_QUERY}"
  "RangeQuery=${RANGE_QUERY}"
  "AllowAdhocQuery=${ALLOW_ADHOC_QUERY}"
  "RescoreLookbackDays=${RESCORE_LOOKBACK_DAYS}"
  "TripMaxDownloadWorkers=${TRIP_MAX_DOWNLOAD_WORKERS}"
  "WarmPoolSize=${WARM_POOL_SIZE}"
  "WarmupRate=${WARMUP_RATE}"
  "LogRetentionDays=${LOG_RETENTION_DAYS}"
  "AlarmEmail=${ALARM_EMAIL}"
  "InvokerPrincipalArns=${INVOKER_PRINCIPAL_ARNS}"
  "ReservedConcurrency=${RESERVED_CONCURRENCY}"
  "FlagThreshold=${FLAG_THRESHOLD}"
  "InferenceBatchSize=${INFERENCE_BATCH_SIZE}"
  "MaxDownloadWorkers=${MAX_DOWNLOAD_WORKERS}"
  "WindowSize=${WINDOW_SIZE}"
  "ImagenetNormalize=${IMAGENET_NORMALIZE}"
  "PgHost=${PG_HOST}"
  "PgPassword=${PG_PASSWORD}"
  "PgPort=${PG_PORT}"
  "PgDatabase=${PG_DATABASE}"
  "PgUser=${PG_USER}"
  "SourcePgHost=${SOURCE_PG_HOST}"
  "SourcePgDatabase=${SOURCE_PG_DATABASE}"
  "SourcePgUser=${SOURCE_PG_USER}"
  "SourcePgPassword=${SOURCE_PG_PASSWORD}"
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
