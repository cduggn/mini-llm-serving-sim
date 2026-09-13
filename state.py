"""The one request shape shared by admission, routing and the harness.

This is lifted unchanged from the live gateway it was written against, so the
same `should_shed` runs against a real service and against the simulator here.
Only the fields the simulation reads are documented; `body`, `messages` and
`token_ids` are carried so the dataclass stays compatible with the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingRequest:
    # Normalized request shared by admission, queueing, routing, and forwarding.
    # n_in is estimated input size; n_out is the requested maximum output.
    # body retains the original API payload; token_ids serve local decisions.
    n_in: int
    n_out: int
    deadline_s: float
    tenant: str
    token_ids: list[int]
    messages: list
    stream: bool
    body: dict
    deadline_at: float
    priority: int | None = None
    deadline_ms: int = 2000
    request_class: str = "interactive"
