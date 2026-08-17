# Incident Note: Opal Relay Sent the Same Alert 186 Times

*Fictional incident note: Opal Relay and all operational data in this document are invented.*

At 09:12 on a fictional Tuesday, our delivery monitor began sending “route delayed” alerts to 62 warehouse supervisors. By 09:19, it had produced 186 messages. No shipment was actually late. The incident ended at 09:41 after the on-call engineer disabled rule `ALERT-7C`.

The immediate cause was a retry worker that treated a slow response as a failed response. The notification service usually answered within 400 milliseconds, but a maintenance query raised the median to 2.8 seconds. Our worker waited two seconds, assumed failure, and tried again. Because the first request had succeeded, each retry created a duplicate.

> "Timeout does not mean the work did not happen."

For example, event `INC-204` entered the queue once at 09:14. Three workers selected it after separate timeouts, and all three sent the same alert. A simple idempotency check would have identified the repeated event before delivery. We did have such a check for invoices, but not for operational notices because those notices were considered low risk.

That assumption was incorrect, though the impact stayed narrow. Supervisors ignored later alerts for about an hour, and the fictional support desk spent $860 of staff time confirming that routes were normal. We detected no lost orders and no exposed customer data. The 37% rise in help-desk calls is an association, not a precise measure of trust; a scheduled inventory change happened that morning too.

We made three changes. First, every notice now requires a stable event key. Second, the worker records success before acknowledging the queue item. Third, a circuit breaker stops a template after five same sends to one recipient in ten minutes. We will not reduce the timeout alone, because a different delay could recreate the issue.

The primary follow-up is a 30-day check of duplicates, suppressed sends, and operator response. A zero count will be encouraging but not sufficient proof that the design is correct under every load pattern.

The fictional timeline is at https://opalrelay.example/incidents/INC-204. Reports would go to ops@opalrelay.example; @OpalRelay owns the review.
