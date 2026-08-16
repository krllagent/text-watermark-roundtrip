# Product Memo for the Quiet Hours Pilot

This memo proposes a four-week pilot for Quiet Hours, a fictional scheduling feature in the Cedar Desk app. The feature would let a team select a period when routine notifications wait in a private queue. Urgent messages would still arrive. The primary goal is to reduce avoidable interruption without hiding work that requires fast action.

The problem came from interviews inside the fictional company. People often mute every channel when they need focus, then forget to restore one of them. The result is uneven: a low-impact comment waits beside a severe customer issue, and neither sender knows what happened. Quiet Hours offers a simple method. A user chooses a time window, the app keeps routine alerts, and a summary appears afterward.

The pilot should begin with twelve volunteers from two internal teams. Participants will use a specific rule for urgency: an alert may bypass the queue only when someone needs action within thirty minutes. The sender must choose the bypass and add a short justification. This extra action should lower casual escalation while retaining an explicit path for time-sensitive work.

During week zero, the team will document notification routes and ask volunteers to confirm their working hours. That baseline prevents seasonal deadlines from distorting the comparison. Support staff will receive a paper escalation guide, and the pilot owner will review bypass logs each morning.

The first version needs three controls. A participant can alter or finish a window at any time. A team administrator can cancel a scheduled window after informing the participant. The system must display which messages were held and which bypassed the queue. These controls are necessary because a hidden queue would introduce a trust problem, even if delivery remained technically valid.

We will collect event counts without message content. The useful measures are windows started, windows completed, alerts held, bypasses, early endings, and summaries opened. We will also ask one weekly prompt: “Did Quiet Hours protect a focus block for you today?” The rating scale runs from one to five. No free text is required, and the synthetic contact for the study is quiet-hours@example.com.

The success threshold is intentionally conservative. At least 70% of scheduled windows should run to completion without an early stop. Routine alerts should fall by 15% during those windows, compared with each participant's own baseline. A majority of participants should choose four or five on the weekly query. These figures are decision rules for this pilot, not a prediction about a larger population.

Several risks require a thorough review. Teams may label too many messages urgent. A time-zone error may start a window at the wrong hour. A delayed summary may leave held work difficult to find. We can spot the first issue through bypass counts, catch the second with schedule tests, and verify the third by tracing the summary job. The pilot should pause if any participant misses a genuine incident because of the feature.

The version is deliberately basic. It includes one recurring window, one bypass action, and one summary page. It does not include shared calendars, automatic priority scoring, or manager reports. Those concepts may become useful later, but adding them now would leave the evidence harder to interpret. A narrow version gives the team enough evidence to choose the next direction.

The budget limit is $600 in fictional internal time, with no paid services. If the pilot meets its rules and participants want to preserve it, the next step is a larger test across four teams. If it fails, we should keep the event record, inform the volunteers, and delete the feature flag. Either result can improve the product decision. The pilot exists to resolve one well-defined question, not to defend a proposal already chosen.
