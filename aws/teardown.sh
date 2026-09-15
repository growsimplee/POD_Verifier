#!/usr/bin/env bash
# Destroy one POD scoring environment so it can be rebuilt from a clean merge.
#
#   Destroys the AWS resources only. pod_scores is sarathy's table, created by
#   sarathy's Flyway migration V198 — this project has no database access and
#   leaves the scored data untouched.
#
#   1. (nothing — the database belongs to sarathy)
#   2. delete the CloudFormation stack        (function, roles, schedules, DLQ,
#                                              alarms, SNS topic, log group)
#   3. delete the ECR repository and images
#   4. delete any log group the stack did not own
#
# Usage:
#   AWS_REGION=us-east-2 STAGE=stg STACK_NAME=pod-scoring-stg ./teardown.sh
#
#   KEEP_ECR=true        skip step 3 — keep the repository and its images
#   ASSUME_YES=true      skip the typed confirmation (for scripted use only)
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AWS_REGION="${AWS_REGION:-us-east-2}"
STAGE="${STAGE:-stg}"
STACK_NAME="${STACK_NAME:-pod-scoring-${STAGE}}"
ECR_REPOSITORY="${ECR_REPOSITORY:-pod-pipeline}"
LAMBDA_FUNCTION="${LAMBDA_FUNCTION:-pod-pipeline-${STAGE}}"
LOG_GROUP="/aws/lambda/${LAMBDA_FUNCTION}"
KEEP_ECR="${KEEP_ECR:-false}"
ASSUME_YES="${ASSUME_YES:-false}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

cat <<EOF

  About to PERMANENTLY DESTROY the POD scoring environment:

    account          ${ACCOUNT_ID}
    region           ${AWS_REGION}
    stage            ${STAGE}
    stack            ${STACK_NAME}
    function         ${LAMBDA_FUNCTION}
    ECR repository   ${ECR_REPOSITORY}            $([ "${KEEP_ECR}" = "true" ] && echo "(KEPT)")
    log group        ${LOG_GROUP}
    database         untouched — pod_scores is sarathy's (V198)

EOF

if [[ "${ASSUME_YES}" != "true" ]]; then
  read -r -p "Type the stack name to confirm: " reply
  if [[ "${reply}" != "${STACK_NAME}" ]]; then
    echo "Got '${reply}', expected '${STACK_NAME}'. Nothing was deleted." >&2
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# 1. Database — NOT ours to drop
# --------------------------------------------------------------------------- #
# pod_scores lives in sarathy's schema and is created by sarathy's Flyway
# migration V198__pod_scores.sql. This project no longer has database access,
# so tearing it down here is neither possible nor correct: dropping the table
# would leave sarathy's migration history claiming it exists. If the data
# really must go, do it from sarathy with a forward migration.
echo "==> [1/4] pod_scores belongs to sarathy (V198) — nothing to drop from here"

# --------------------------------------------------------------------------- #
# 2. CloudFormation stack
# --------------------------------------------------------------------------- #
if aws cloudformation describe-stacks \
     --stack-name "${STACK_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> [2/4] deleting stack ${STACK_NAME} (a few minutes — VPC ENIs are slow)"
  aws cloudformation delete-stack --stack-name "${STACK_NAME}" --region "${AWS_REGION}"
  if aws cloudformation wait stack-delete-complete \
       --stack-name "${STACK_NAME}" --region "${AWS_REGION}"; then
    echo "    stack deleted"
  else
    echo "    stack delete did not complete — recent failures:" >&2
    aws cloudformation describe-stack-events \
      --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
      --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId,ResourceStatusReason]" \
      --output table >&2 || true
    exit 1
  fi
else
  echo "==> [2/4] stack ${STACK_NAME} not found — nothing to delete"
fi

# --------------------------------------------------------------------------- #
# 3. ECR repository (images included)
# --------------------------------------------------------------------------- #
if [[ "${KEEP_ECR}" == "true" ]]; then
  echo "==> [3/4] KEEP_ECR=true — leaving ${ECR_REPOSITORY} in place"
elif aws ecr describe-repositories \
       --repository-names "${ECR_REPOSITORY}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> [3/4] deleting ECR repository ${ECR_REPOSITORY} and every image in it"
  aws ecr delete-repository \
    --repository-name "${ECR_REPOSITORY}" --region "${AWS_REGION}" --force >/dev/null
  echo "    repository deleted"
else
  echo "==> [3/4] ECR repository ${ECR_REPOSITORY} not found in ${AWS_REGION}"
fi

# --------------------------------------------------------------------------- #
# 4. Log group — only orphans; the stack owns its own now
# --------------------------------------------------------------------------- #
if aws logs describe-log-groups \
     --log-group-name-prefix "${LOG_GROUP}" --region "${AWS_REGION}" \
     --query "logGroups[?logGroupName=='${LOG_GROUP}'] | length(@)" --output text | grep -qv '^0$'; then
  echo "==> [4/4] deleting leftover log group ${LOG_GROUP}"
  aws logs delete-log-group --log-group-name "${LOG_GROUP}" --region "${AWS_REGION}"
else
  echo "==> [4/4] no leftover log group"
fi

cat <<EOF

  Done. ${STAGE} is gone from ${AWS_REGION}.

  Left alone on purpose — these belong to the platform, not to this service:
    · the VPC, its subnets and NAT
    · the RDS instance, the sarathy database and the kaptaan table
    · the CircleCI contexts and their credentials
    · Sarathy's aws.lambda.podverificationlambda setting

  To rebuild: merge to stag-main and let CI run.

EOF
