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
| `score-one-trip-with-links` | Same, but supplies the POD links directly. Use to test against a specific image. Supply `awb` too, or the handler still calls sarathy to resolve it. |
| `backfill-date-range` | Re-scores every trip whose POD was captured in a date range. |
| `backfill-today` | Today's PODs — the empty event, and what the disabled daily schedule sends. |
| `warmup` | Loads the model and returns. Scores nothing, writes nothing. Safe anywhere. |

Re-running any scoring event is safe: sarathy reports which links already carry
a score and those are not re-downloaded, and the write upserts on
`(awb, podLink, runDate)`.

## Two things to know before clicking Test

**The console invokes synchronously.** It waits for the response, so a
`backfill-date-range` can hold the tab for up to the 15-minute function timeout,
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

**No schema events.** There used to be a `migrate-apply-schema` and a matching
destructive drop. `pod_scores` is sarathy's table now, created by sarathy's
Flyway migration `V198__pod_scores.sql`, and this function has no database
access at all — so neither event has anything to do.

**No ad-hoc SQL event.** The scorer cannot run a query; it asks sarathy for a
trip or a date range. If a backfill needs a different selection than "captured
between these dates", that belongs in sarathy's
`GET /internal/pod-scoring/trips`, in review, rather than in a payload pasted
into a console field.

## Checking the result

The scores land in sarathy's database, so read them from sarathy or Metabase —
this function has no connection of its own:

```sql
SELECT status, count(*), max(scored_at)
FROM pod_scores WHERE run_date = CURRENT_DATE GROUP BY status;

SELECT * FROM pod_scores WHERE trip_id = '12345' ORDER BY scored_at DESC;
```
