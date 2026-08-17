# Design Critique: The Waveform Looked Precise and Explained Too Little

*Fictional design critique: Hushnote, its prototype, and all study results were invented for this holdout.*

Hushnote tested a mobile screen that summarizes short voice memos. The design placed a large audio waveform above a three-line text summary. It looked technical and reassuring. In practice, the waveform occupied 38% of the screen while answering none of the questions people asked most often: who spoke, whether the note was complete, and where uncertainty remained.

Twelve fictional participants reviewed six memos each. Ten tapped the waveform expecting to select a sentence, though it was only decorative. Seven missed the “summary may omit details” notice beneath the fold. The task completion rate was 68%, and the prototype cost $3,600 to build.

> "Precision in appearance is not the same as precision in meaning."

One participant, Leni, offers a clear example. A memo said, “send the blue sample, not the green one.” The summary displayed “send the sample,” while the waveform showed two strong peaks. Leni assumed the peaks marked color choices and selected blue by luck. The answer was correct, but the method was unreliable.

We should remove the large waveform from the summary screen and use token `OMISSION_FLAG` beside any sentence with low source coverage. A small play control can retain access to audio. The main action should be “check against recording,” not “accept summary.”

There is an unresolved relation between visual confidence and user confidence. Eight participants rated the original screen as trustworthy, yet five of those eight also made at least one detail error. The waveform may have raised confidence because it resembled an instrument, or people who already trusted audio software may simply have liked the waveform. This study cannot identify the direction.

The next prototype will compare a plain transcript-first layout with a compact summary-first layout. We will not optimize for speed alone. Success requires at least 85% correct detail selection, no increase in missed negation, and a median review time below 90 seconds.

The fictional critique is at https://hushnote.example/design/waveform. Comments would go to design@hushnote.example; @HushnoteUX owns the revision. The team has a further $2,100, which is sufficient for one moderated round, not a full redesign.
