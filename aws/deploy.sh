#!/usr/bin/env bash
# Build the POD scorer image → create the ECR repo if absent → push.
# NO SAM — plain Docker + AWS CLI. Infra is CloudFormation (aws/infra/stack.yaml)
# applied by aws/provision-stack.sh, which is what actually points the function
# at an image; this script only has to get the image into ECR.
#
# Handler: event-driven per-trip scorer (Sarathy invokes it when a rider raises a
# POD verification request), plus batch modes for backfills. The function holds
# no database credentials — it reads rows from and writes scores back to
# sarathy's /internal/pod-scoring API, which owns the pod_scores table.
#
# Everything here is idempotent, so CI can run it on a completely empty account:
#   * the ECR repository is created when missing
#   * the Lambda update is skipped automatically when the function does not exist yet
#
# Local build smoke-test without AWS credentials:
#   DRY_RUN=true ./deploy.sh
#
# Push only, never touch the function (CFN will set the image):
#   SKIP_LAMBDA_UPDATE=true ./deploy.sh
#
set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${AWS_DIR}/lambda_scorer"

# The trained checkpoint is baked into the image — fail fast if it's missing.
if [[ ! -f "${CTX}/model/best.pt" ]]; then
  echo "ERROR: ${CTX}/model/best.pt not found. Commit/copy the trained checkpoint first." >&2
  exit 1
fi

AWS_REGION="${AWS_REGION:-us-east-2}"
STAGE="${STAGE:-stg}"
ECR_REPOSITORY="${ECR_REPOSITORY:-pod-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_NAME="${LOCAL_NAME:-pod-pipeline-local}"
SKIP_LAMBDA_UPDATE="${SKIP_LAMBDA_UPDATE:-false}"
DRY_RUN="${DRY_RUN:-false}"

LAMBDA_FUNCTION="${LAMBDA_FUNCTION:-pod-pipeline-${STAGE}}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "==> DRY_RUN: docker build only (${CTX}) linux/amd64"
  docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --load \
    -t "${LOCAL_NAME}:${IMAGE_TAG}" \
    "${CTX}"
  echo "DRY_RUN OK: local image ${LOCAL_NAME}:${IMAGE_TAG}"
  exit 0
fi

if [[ -n "${AWS_ACCOUNT_ID:-}" ]]; then
  :
else
  if ! AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"; then
    echo "aws sts get-caller-identity failed; set AWS_ACCOUNT_ID or fix credentials / AWS_PROFILE." >&2
    exit 1
  fi
fi

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REMOTE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

# Create the repository on first run. ECR is regional, so a brand-new region
# (e.g. standing prod up in ap-south-1) always lands here.
if aws ecr describe-repositories --repository-names "${ECR_REPOSITORY}" \
     --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> ECR repository ${ECR_REPOSITORY} already exists"
else
  echo "==> creating ECR repository ${ECR_REPOSITORY} in ${AWS_REGION}"
  aws ecr create-repository \
    --repository-name "${ECR_REPOSITORY}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE >/dev/null
  # Keep the registry from growing without bound: hold the last 15 images.
  aws ecr put-lifecycle-policy \
    --repository-name "${ECR_REPOSITORY}" \
    --region "${AWS_REGION}" \
    --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"keep last 15","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":15},"action":{"type":"expire"}}]}' >/dev/null
fi

echo "==> ECR login ${REGISTRY}"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "==> docker buildx (${CTX}) linux/amd64 (same as reference deploy.sh)"
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -t "${LOCAL_NAME}:${IMAGE_TAG}" \
  "${CTX}"

docker tag "${LOCAL_NAME}:${IMAGE_TAG}" "${REMOTE_URI}"

echo "==> docker push ${REMOTE_URI}"
docker push "${REMOTE_URI}"

echo "Image pushed to ECR!"
echo "Image URI: ${REMOTE_URI}"

if [[ "${SKIP_LAMBDA_UPDATE}" == "true" ]]; then
  echo "SKIP_LAMBDA_UPDATE=true → skipping lambda update-function-code."
elif ! aws lambda get-function-configuration \
       --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "${LAMBDA_FUNCTION} does not exist yet → image-only push."
  echo "provision-stack.sh will create it pointing at ${REMOTE_URI}."
else
  echo "==> aws lambda update-function-code ${LAMBDA_FUNCTION}"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --image-uri "${REMOTE_URI}" \
    --region "${AWS_REGION}"
fi

echo "Done."
