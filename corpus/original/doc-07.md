# Incident Review: The Duplicate Ticket Export

On 12 May, the fictional Northwind Workshop exported customer support tickets twice to its internal archive. No messages left the company, and no customer action was required. The duplicate run created 418 extra files and delayed the morning report by forty-six minutes. This review explains the failure mechanism, the remediation, and the work needed to reduce the chance of a recurrence.

At 07:00, a scheduled task began the routine export. The task selected tickets changed since the prior run, produced one file per ticket, and stored a completion record. At 07:09, the archive server returned a timeout after accepting the final batch. The exporter treated the timeout as a failure and tried the complete job again. It lacked a deduplication safeguard for files already present.

At 07:21, an analyst noticed a sudden surge in the archive count. She checked two filenames and found matching ticket identifiers with different run suffixes. She informed the incident channel and paused the reporting job. This conservative choice kept duplicate data out of the summary while preserving the files for inspection. The first direct message was: "Archive export paused; duplicate files under review."

At 07:28, the coordinator wrote a minimal containment plan. One person would verify the network boundary, one would locate the earliest valid file, and one would keep a minute-by-minute record. This role split helped the group start quickly without making parallel edits. It also gave each later finding a traceable source.

The team began with three questions. Had any file been sent outside the archive? Did the second run alter an original? Could the duplicates be removed without deleting unique tickets? Log review gave a definitive conclusion on the first two queries: all writes stayed on the internal server, and filenames prevented overwrites. A digest review verified the third point for a sample, then a complete comparison confirmed it.

The primary cause was an absent idempotency key at the archive boundary. The exporter created a new run suffix after every retry, so the storage service saw each file as a separate object. A secondary issue made detection slower. The dashboard displayed successful batches but did not report the number of files produced per ticket. Both systems behaved according to their basic rules, yet their combined behavior was incorrect.

At 07:38, the team selected a recovery script that matched files by ticket identifier and digest. It marked the later copy for review but did not erase anything automatically. Two operators examined the report, retained each earliest file, and deleted only exact duplicates. The cleanup finished at 08:02. A final count and random sample produced a complete archive, after which reporting resumed.

The immediate fix added a stable operation identifier to every file request. The archive now returns the earlier reply when it receives a matching identifier and digest. A distinct digest paired with a reused identifier becomes a non-retryable error. The team also changed the exporter to keep its identifier across retries. These two controls address the main problem at both sides of the interface.

The follow-up work has a focused objective. First, add a metric for files per ticket and configure an alert when it rises above one. Second, attempt a timeout after storage accepts a batch, then verify that a retry produces no extra file. Third, document a manual recovery procedure at https://northwind.invalid/runbooks/export. The synthetic owner is archive-owner@example.com, and the tracker label is #ExportReview.

This incident had a limited operational cost and no confirmed external effect. That narrow impact should not lower the priority of the fix. That weakness in billing or delivery could have a severe impact. The useful result is a narrower contract: one ticket identifier and one digest should produce one stored object. The team can now test that statement directly instead of relying on an inference from a timeout.
