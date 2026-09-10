#!/usr/bin/env bash
# Destroy one POD scoring environment so it can be rebuilt from a clean merge.
#
#   DROPS THE pod_scores TABLE AND ITS VIEW. Every score computed so far is gone.
#   There is no undo and no backup taken here.
#
# Order matters. The database lives in a private subnet that this shell probably
# cannot reach, so the Lambda drops the table for us — which means the drop has
# to happen while the function still exists, i.e. before the stack is deleted.
#
#   1. drop pod_scores + pod_scores_flagged   (via the Lambda, inside the VPC)
#   2. delete the CloudFormation stack        (function, roles, schedules, DLQ,
#                                              alarms, SNS topic, log group)
#   3. delete the ECR repository and images
#   4. delete any log group the stack did not own
#
# Usage:
#   AWS_REGION=us-east-2 STAGE=stg STACK_NAME=pod-scoring-stg ./teardown.sh
#
#   KEEP_DATABASE=true   skip step 1 — leave pod_scores intact
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
KEEP_DATABASE="${KEEP_DATABASE:-false}"
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
    database         DROP pod_scores, pod_scores_flagged   $([ "${KEEP_DATABASE}" = "true" ] && echo "(KEPT)")

EOF

if [[ "${ASSUME_YES}" != "true" ]]; then
  read -r -p "Type the stack name to confirm: " reply
  if [[ "${reply}" != "${STACK_NAME}" ]]; then
    echo "Got '${reply}', expected '${STACK_NAME}'. Nothing was deleted." >&2
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# 1. Database — while the function that can reach it still exists
# --------------------------------------------------------------------------- #
if [[ "${KEEP_DATABASE}" == "true" ]]; then
  echo "==> [1/4] KEEP_DATABASE=true — leaving pod_scores in place"
elif ! aws lambda get-function-configuration \
       --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> [1/4] ${LAMBDA_FUNCTION} does not exist — cannot drop the table from here."
  echo "          If pod_scores still exists, drop it manually:"
  echo "            DROP VIEW IF EXISTS pod_scores_flagged; DROP TABLE IF EXISTS pod_scores;"
else
  echo "==> [1/4] dropping pod_scores via ${LAMBDA_FUNCTION}"
  # confirm must equal the function's own name — a payload copied from another
  # environment cannot drop this one.
  aws lambda invoke \
    --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" \
    --payload "{\"migrate\": \"drop\", \"confirm\": \"${LAMBDA_FUNCTION}\"}" \
    --cli-binary-format raw-in-base64-out /tmp/pod-teardown-drop.json >/dev/null
  cat /tmp/pod-teardown-drop.json; echo
  if ! grep -q '"status": "dropped"' /tmp/pod-teardown-drop.json; then
    echo "Drop did not report success. Stopping so the stack stays up and you can retry." >&2
    exit 1
  fi
fi

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
