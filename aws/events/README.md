# Console test events

Payloads for invoking the scorer from the Lambda console, so a run does not
require a terminal or AWS credentials on your laptop — just console access and
`lambda:InvokeFunction`.

## Saving them

Do this once per environment. Use **shareable** events so the team sees the same
set instead of everyone keeping private copies.

1. Lambda console → `pod-pipeline-stg` → **Test** tab
2. **Create new event** → set *Event sharing settings* to **Shareable**
3. Name it after the file (`score-one-trip`), paste the file's contents, **Save**
4. Repeat for the others

They then appear in the **Test** dropdown for anyone with access to the function.

## What each one does

| Event | Effect |
|---|---|
| `score-one-trip` | Scores one trip. **Edit `trip_id` before running.** The live path — identical to what Sarathy sends. |
| `score-one-trip-with-links` | Same, but supplies the POD links directly. Use when the `kaptaan` row does not exist yet, or to test against a specific image. |
| `backfill-date-range` | Re-scores everything in a date range, via `RANGE_QUERY`. |
| `backfill-query` | Re-scores whatever a read-only `SELECT` returns. |
| `backfill-full-sweep` | The whole `SOURCE_QUERY` dataset — what the disabled daily schedule sends. |
| `warmup` | Loads the model and returns. Scores nothing, writes nothing. Safe anywhere. |
| `migrate-apply-schema` | Applies `infra/schema.sql`. Idempotent — CI runs it on every deploy. |

Re-running any scoring event is safe: links already scored within
`RESCORE_LOOKBACK_DAYS` are skipped, and writes upsert on
`(awb, pod_link, run_date)`.

## Two things to know before clicking Test

**The console invokes synchronously.** It waits for the response, so a
`backfill-full-sweep` can hold the tab for up to the 15-minute function timeout,
and the browser may give up before the run does — the run still completes. For
anything batch-sized, prefer an async CLI invoke and watch the logs:

```bash
aws lambda invoke --function-name pod-pipeline-stg --region us-east-2 \
  --invocation-type Event --cli-binary-format raw-in-base64-out \
  --payload file://backfill-date-range.json /dev/null

aws logs tail /aws/lambda/pod-pipeline-stg --region us-east-2 --since 15m --follow
```

**These are live.** A test event against `pod-pipeline-prod` scores production
trips and writes production rows. Check which function is selected first — the
payloads are identical between environments, which is convenient and also the
easy way to run the wrong one.

## What is deliberately not here

There is no test event for the destructive schema drop
(`{"migrate": "drop", …}`). It is a one-click action in a dropdown next to
routine ones, and the whole point of the confirmation token is that dropping the
table should be awkward. `aws/teardown.sh` builds that payload when it needs it.

## Checking the result

```sql
SELECT status, count(*), max(scored_at)
FROM pod_scores WHERE run_date = CURRENT_DATE GROUP BY status;

SELECT * FROM pod_scores WHERE trip_id = '12345' ORDER BY scored_at DESC;
```
