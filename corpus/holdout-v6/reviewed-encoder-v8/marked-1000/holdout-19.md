# Launch Note: The Fictional Status Page Is Deliberately Boring

*Fictional launch note: Cloudberry Desk, every service, and all incident figures below are invented.*

Today Cloudberry Desk is launching a public status page for its fictional scheduling service. It has four components, three states, and no animated uptime score. That small scope is intentional.

During six invented incidents last quarter, support received 428 messages asking whether the service was down. The first accurate public update took a median of 34 minutes because engineers wrote a fresh explanation for every channel. The new method lets the incident lead select component, state, and a short note using template `STATUS_V1`.

> "A status page should reduce uncertainty, not advertise reliability."

For example, if calendar sync is delayed but bookings still save, the page will display “Calendar sync: degraded” and keep “Booking: operational.” It will not mark the entire service unavailable. Updates must state what users can do now, such as retry after 15 minutes or continue without sync.

Our fictional build cost $4,800. In a tabletop exercise, the first update appeared in 6 minutes instead of 34, an 82.4% reduction. That result does not show how the team will perform during a stressful real incident. The exercise had a known start time, and the incident lead was already at a computer.

There is another ambiguous measure: fewer support messages might mean clearer communication, or it might mean customers stopped looking for help. We will pair message volume with page visits and a one-question usefulness response. A low message count alone is not success.

The page will retain 90 days of incident history. It will not publish customer names, internal hostnames, speculative causes, or an estimated recovery time without an owner. @CloudberryOnCall can post an initial scoped update; a second person must verify any claim that data were lost or exposed.

For the first month, the main target is an accurate update within ten minutes for every declared incident and a correction log for any altered statement. We expect the page to look plain. Its job is to tell the truth quickly enough to help.

The fictional page is https://status.cloudberrydesk.example. Feedback would go to status@cloudberrydesk.example, and the monthly operating ceiling is $180.
