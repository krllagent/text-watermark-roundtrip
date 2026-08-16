# Decision Memo: Replace the Visitor Sign-In Binder

This memo concerns Red Cedar House, a fictional shared office. Every person, organization, address, and figure is invented. The decision is whether to replace a paper visitor binder with a simple local tablet form for a three-month trial.

The main problem is incomplete departure records. Visitors write an arrival time, but hosts often forget to add an exit time. The evening coordinator must verify the occupied-room list and update the binder by hand. A clear and accurate current list would allow the coordinator to identify who may still be inside.

The proposed method uses a small tablet at reception. A visitor will select a fictional host, enter a first name and arrival purpose, and receive a temporary badge. The host will record departure on the reception tablet. The system will delete visitor details after seven days and retain only daily counts for the trial report.

The paper option has a low direct cost and no battery. It is also easy to explain. Its primary weakness is that one line can reveal another visitor's name. Staff can shred old pages on a documented weekly schedule, but that step does not solve exposure on the current page. The handwriting can also be hard to read.

The tablet option can display one entry at a time and hide prior records. It can show a late departure alert and record an accurate audit timestamp. The device adds a different risk: a frozen screen could block entry. The desk therefore needs a blank backup sheet and a plain recovery instruction.

Three alternatives were considered. A wall camera was rejected because it would collect more information than the goal requires. A staffed desk was rejected because the fictional office cannot cover twelve hours. A hosted visitor platform was rejected because the trial does not need remote access or a big feature set.

The trial has a clear fictional budget of $450, including a 15% replacement allowance. Configuration notes can live at https://example.test/red-cedar/sign-in, while synthetic support mail goes to visitors@example.com. Internal test messages may include @redcedar_desk and #sign-in-trial. The displayed notice says “Details are removed after seven days.”

Success needs an accurate definition. The tablet should record a departure for at least 95% of visits, expose no prior visitor entry during usual operation, and recover from a simulated restart within three minutes. Staff time should not grow by more than ten minutes per day. These thresholds are sufficient for a trial decision.

The coordinator will run a basic weekly review. They will compare the host list with departure records, verify deletion, and record any support question. They should also detect workarounds, such as hosts sharing one badge. A thorough review must include inconvenient results alongside successful days.

A small paper survey will ask reception staff whether the notice is clear and whether the backup is easy to find. The coordinator will check responses once per week and record the number invited. That count prevents a few vocal comments from standing in for the whole trial.

The largest uncertainty is host behavior. A powerful reminder can still be ignored. The team will start with one notification at departure and avoid repeated prompts. If the record stays incomplete, the likely cause may be the handoff rather than the tool. That outcome would argue against further software work.

My recommendation is to begin the local tablet trial and keep the binder as an emergency backup. Do not run both as equal records, because duplicate records can diverge. The tablet is the source of truth during normal operation. The paper sheet is used only during a documented outage.

After three months, the office should start its main review and choose among adoption, revision, and removal. The final response should depend on observed departure coverage, privacy checks, staff time, and failure recovery. A positive result would support this narrow setup. It would not prove the same approach is appropriate for every office.
