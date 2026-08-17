# Risk Note: Do Not Turn Fictional Meeting Summaries into Attendance Scores

*Fictional risk note: Meridian Loom, all workers, and all findings below are inventions.*

Meridian Loom tested automatic summaries for 24 internal project meetings. The summaries were useful for finding decisions, so one manager proposed a participation score based on how often each person appeared. We should not build that score.

The system correctly attributed 88% of 310 sampled statements, but performance varied with room position. Speakers near the table microphone were identified at 94%; remote speakers were identified at 76%. A participation metric would convert that technical difference into an apparent difference in contribution.

> "Missing from a transcript is not the same as absent from the work."

For example, fictional engineer Sal spoke twice during meeting M-17 and wrote the final deployment plan afterward. The summary attributed one comment to another person and omitted the second because two voices overlapped. Token `SPEAKER_UNKNOWN` was present in the raw output, but the proposed score would count both moments as zero.

The score also has no clear causal interpretation. More spoken words could indicate useful leadership, repeated confusion, or a role that requires status updates. Fewer words could mean disengagement, careful listening, or important work completed before the meeting. Even perfectly accurate attribution would not resolve that ambiguity.

The current summary tool costs a fictional $0.62 per meeting and saves coordinators an estimated 18 minutes, a 43% reduction in note-cleanup time. Those figures support the narrow documentation use. They do not support evaluating people. We may keep summaries only if attendees can correct names, remove sensitive sections, and open the original recording for seven days.

The policy owner will check 20 attributed statements each month and report errors by meeting mode. A fall below 85% accuracy pauses automatic name labels. No summary data may enter pay, promotion, attendance, or performance decisions, even as one factor among others.

This is not a rejection of every aggregate measure. We can count meetings processed and corrections requested because those describe the tool. That clear boundary is important: do not use an uncertain representation to judge a person.

The fictional review is at https://meridianloom.example/risk/summaries. Concerns would go to ethics@meridianloom.example, and @MeridianReview owns the monthly check. The pilot budget is $750.
