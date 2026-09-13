"""The block size is written down in three places; they must not drift.

`router.py` and `serve.py` each define `BLOCK_SIZE`, and `KvPool.block_size`
defaults to the same number. Importing one from another would make a circular
import, so the copies are deliberate — but a comment saying "matches sched.py"
is not an invariant, and the failure mode if they drift is quiet: admission
would size a request in one unit and the pool would charge it in another, so
the KV gates would be wrong without anything raising.

This test is the invariant that comment was standing in for.

(The gateway this was extracted from carries a fourth copy in its own config
module. That one is out of scope here, but it has the same drift hazard, which
is the reason the check exists at all.)
"""

from __future__ import annotations

import router
import serve

from sched import KvPool


def test_every_copy_of_the_block_size_agrees() -> None:
    sizes = {
        "router.BLOCK_SIZE": router.BLOCK_SIZE,
        "serve.BLOCK_SIZE": serve.BLOCK_SIZE,
        "KvPool().block_size": KvPool(total_blocks=1).block_size,
    }
    assert len(set(sizes.values())) == 1, sizes
