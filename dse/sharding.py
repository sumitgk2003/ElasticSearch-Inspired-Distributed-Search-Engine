"""Document-to-shard routing: shard = hash(doc_id) mod N."""
from __future__ import annotations

import zlib


def shard_for(doc_id: str, num_shards: int) -> int:
    # Python's built-in hash() is salted per process, so use a stable hash.
    return zlib.crc32(doc_id.encode("utf-8")) % num_shards
