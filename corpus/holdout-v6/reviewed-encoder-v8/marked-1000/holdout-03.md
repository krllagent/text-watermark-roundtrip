# Why a Fictional Job Runner Uses a Receipt Before It Uses a Retry

*Fictional technical explainer: Bramble Grid, its traffic, and its costs exist only in this test document.*

A retry sounds simple: if a job does not finish, run it again. The difficult part is that a client can lose the answer after the server has completed the work. Repeating the request may then produce two correct executions and one incorrect business outcome.

Bramble Grid, a fictional image-processing service, handles this with a receipt. Before a worker begins an export, it stores a `job_key`, the requested operation, and a blank result field in one transaction. A later worker with the same key does not start another export. It either returns the retained result or waits for the first worker to finish.

> "A retry asks again; a receipt remembers why."

Consider job `elm-4821`. The client sends a request at 10:00:00, and the export finishes 11 seconds later. The network closes before the response arrives. At 10:00:15, the client tries again. Without a receipt, the service creates a second 240 MB archive and charges another fictional $0.004. With the receipt, it displays the first archive reference.

This method reduced duplicate exports from 73 to 4 in a fictional 8,000-job load test, a 94.5% decrease. It did not remove every duplicate. Four older clients generated a fresh key on each attempt, so the server could not identify the relation between requests. That caveat matters: idempotency is a contract shared by client and server, not a server feature in isolation.

There is also an ambiguous edge. If two requests carry the same key but different crop settings, are they duplicates or a collision? Bramble Grid chooses a cautious response: it rejects the second request and shows both parameter hashes. Automatically keeping the first would be fast, but it could hide a caller error.

Receipts add storage and a cleanup question. We retain completed records for 36 hours, which was enough for observed retries, but the window is not universally correct. The primary goal is to make uncertainty explicit: every repeated job should have a clear result, a clear conflict, or a clear expiry.

The fictional protocol is at https://bramblegrid.example/idempotency. Notes would go to docs@bramblegrid.example, and #BrambleNotes tracks changes. The stated completion target is 99.2% within 20 seconds.
