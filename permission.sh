ACCT=895469091637
REGION=us-east-2

check() {
  printf '%-34s ' "$1"
  aws iam simulate-principal-policy \
    --policy-source-arn "arn:aws:iam::${ACCT}:user/circleCi" \
    --action-names "$1" --resource-arns "$2" \
    --query 'EvaluationResults[0].EvalDecision' --output text
}

check cloudformation:CreateChangeSet "arn:aws:cloudformation:${REGION}:${ACCT}:stack/pod-scoring-stg/*"
check ecr:CreateRepository           "arn:aws:ecr:${REGION}:${ACCT}:repository/pod-pipeline"
check lambda:CreateFunction          "arn:aws:lambda:${REGION}:${ACCT}:function:pod-pipeline-stg"
check lambda:InvokeFunction          "arn:aws:lambda:${REGION}:${ACCT}:function:pod-pipeline-stg"
check iam:CreateRole                 "arn:aws:iam::${ACCT}:role/pod-pipeline-fn-stg"
check iam:PassRole                   "arn:aws:iam::${ACCT}:role/pod-pipeline-fn-stg"
check logs:CreateLogGroup            "arn:aws:logs:${REGION}:${ACCT}:log-group:/aws/lambda/pod-pipeline-stg"
check sqs:CreateQueue                "arn:aws:sqs:${REGION}:${ACCT}:pod-pipeline-dlq-stg"
check sns:CreateTopic                "arn:aws:sns:${REGION}:${ACCT}:pod-pipeline-alarms-stg"
check cloudwatch:PutMetricAlarm      "arn:aws:cloudwatch:${REGION}:${ACCT}:alarm:pod-pipeline-errors-stg"
check scheduler:CreateSchedule       "arn:aws:scheduler:${REGION}:${ACCT}:schedule/default/pod-scoring-warmup-stg"
check ec2:CreateSecurityGroup        "*"