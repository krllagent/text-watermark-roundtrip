# Decision Memo: Keep Two Routes for the Fictional Paper Kite Trial

*Fictional decision memo: Paper Kite Studio, the trial, and all figures are original inventions.*

Decision: run route A and route B for six more weeks, but do not build route C.

Paper Kite delivers sample books to independent illustrators. In the fictional spring pilot, route A grouped parcels by postcode and left at 08:00. Route B grouped them by promised date and left at 11:30. Route C would add live driver selection through a new scheduling system.

The main result is close. Route A delivered 214 of 230 parcels on the promised day; route B delivered 207 of 220. That is 93.0% versus 94.1%, not a meaningful winner at this scale. Route A cost $6,800, while route B cost $6,210 because it required one fewer early shift.

> "Choose the next test, not the most impressive diagram."

We should retain both routes because they fail in different ways. For example, route A handled a 38-parcel Monday quickly, but seven illustrators were not yet at their studios. Route B reached more recipients, yet its later departure made rural stops vulnerable to afternoon traffic. The timing likely causes part of each pattern, although postcode density and recipient schedules are mixed into the same comparison.

Route C is not justified. The supplier estimates four weeks and $9,400 to build it, before training or support. Live selection might improve utilization by 12%, but that figure comes from a simulation with fixed travel times. It does not show how drivers would respond to frequent changes. A large interface could also make a basic dispatch problem harder to inspect.

For the extension, operations will use token `route=B` only on days with more than 25 promised-date parcels. On other days, route A remains the default. We will check delivery rate, recipient presence, driver overtime, and cost per successful handoff. A route can continue only if it stays above 92% and below $31 per completed delivery.

This decision is narrow. It does not reject dynamic scheduling for a larger network; it says our current evidence is insufficient. The next review should produce a clear answer without committing us to a platform.

The fictional memo lives at https://paperkite.example/memos/routes. Corrections would go to memo@paperkite.example; @PaperKiteOps records the weekly outcome.
