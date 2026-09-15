# Assignment questions

Where each question is answered in the code.

| # | Question | Where |
|---|---|---|
| Q1 | Where do I prevent accepting work that will time out? | `admit._check_queue_wait` (gate 3): queue length × p50 TTFT vs half the deadline |
| Q2 | Where do I protect KV memory? | Before entry: `admit._check_kv_pressure`, `_check_kv_capacity` (gates 4–5). After: `sched._kv_room`, `_reclaim_blocks` |
| Q3 | Where do I prioritise interactive traffic? | `admit._check_tail_latency` (gate 6) sheds batch; `sched._select_priority` orders work; `_select_victim` never evicts upward |
| Q4 | Where do I stop one tenant monopolising? | `admit._check_allowance` (gates 1–2, `429`); `sched._select_drr` shares prefill between tenants |
| Q5 | Where do I preempt a request? | `sched._preempt`, triggered by `_reclaim_blocks`, victim from `_select_victim` |
| Q6 | Where do I exploit shared prefixes? | `router._prefix_then_load` picks the warm worker; `_cached_tokens` feeds admission gate 4 |
| Q7 | Where do I avoid a worker that has gone quiet? | `router._eligible` / `_is_unknown` (H6); `_load_key` sorts an unknown worker as fully loaded |
| Q8 | Where do I avoid bouncing a request around the fleet? | `router._admissible` / `_would_shed` (H4): refuse once with `Shed(503, retry_after=2.0)` |
