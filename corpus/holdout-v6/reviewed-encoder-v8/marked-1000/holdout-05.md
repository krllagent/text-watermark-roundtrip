# Retrospective: We Removed the Most Requested Button

*Fictional product retrospective: Mosslight and every customer, metric, and event below are invented.*

When Mosslight launched its fictional neighborhood garden app, the most requested feature was a “claim plot” button. Twenty-six of 71 beta users asked for it, so we built it first. Three weeks later, we removed it.

The problem was not code quality. The button worked, and `claim_plot()` passed 48 automated checks. The issue was the promise implied by the word “claim.” Garden coordinators used the app to discuss space, but only a local committee could assign it. A fast digital action looked authoritative even when it had no authority.

> "A button can be accurate and still describe the wrong decision."

One example made this clear. A fictional member named Ivo selected plot 12 at 07:40. Another member had already received that plot in a paper meeting. The app showed Ivo a green confirmation, so both people arrived on Saturday expecting the same space. The coordinator resolved the conflict in 18 minutes, but the interface had created confidence without agreement.

During the 21-day beta, 34 claims were recorded and 9 needed manual correction. That 26.5% correction rate was significant enough to stop the flow. It does not prove users disliked self-service; our label and governance model were entangled. The high request count may have reflected a need for visibility rather than a desire for unilateral control.

We replaced the button with “request a plot.” The new method sends a plain summary to the coordinator and displays a pending state. Median completion grew from 46 seconds to 7.2 hours, while disputes fell from 9 in 34 attempts to 1 in 29. Slower was better for this narrow outcome, although a different community with delegated authority might choose the original approach.

The change cost a fictional $2,750, including research and support. We will not add voting, payments, or identity checks yet. The primary goal for the next month is simple: at least 90% of requests should receive a clear answer within one working day.

The fictional review is at https://mosslight.example/retrospective. Feedback would go to garden@mosslight.example, and @MosslightLab owns metric `plot_state=pending`. The follow-up target is 90%.
