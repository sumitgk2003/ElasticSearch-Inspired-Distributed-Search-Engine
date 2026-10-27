"""Shard allocation: place shard copies on nodes and rebalance on membership change.

Rules (as in Elasticsearch's same-shard allocation decider):
* no two copies of one shard live on the same node;
* copies are spread so node loads differ by at most one where possible;
* when the primary's node leaves, a surviving replica is promoted.

An allocation maps shard -> ordered list of nodes; index 0 is the primary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Allocation = Dict[int, List[str]]


@dataclass
class RebalancePlan:
    allocation: Allocation
    added: List[Tuple[int, str]] = field(default_factory=list)    # (shard, node) copies to start
    removed: List[Tuple[int, str]] = field(default_factory=list)  # (shard, node) copies to stop
    promoted: List[Tuple[int, str]] = field(default_factory=list) # (shard, new primary node)
    unassigned: List[int] = field(default_factory=list)           # shards short of copies


def _loads(alloc: Allocation, nodes: Sequence[str]) -> Counter:
    c = Counter({n: 0 for n in nodes})
    for copies in alloc.values():
        c.update(copies)
    return c


def _place_missing(alloc: Allocation, nodes: Sequence[str], copies_wanted: int,
                   plan: RebalancePlan) -> None:
    loads = _loads(alloc, nodes)
    for shard in sorted(alloc):
        while len(alloc[shard]) < copies_wanted:
            options = [n for n in nodes if n not in alloc[shard]]
            if not options:
                plan.unassigned.append(shard)
                break
            # least-loaded node; node order breaks ties deterministically
            target = min(options, key=lambda n: (loads[n], nodes.index(n)))
            alloc[shard].append(target)
            loads[target] += 1
            plan.added.append((shard, target))


def allocate(num_shards: int, replicas: int, nodes: Sequence[str]) -> Allocation:
    """Initial placement of ``num_shards`` primaries with ``replicas`` replicas each."""
    alloc: Allocation = {s: [] for s in range(num_shards)}
    plan = RebalancePlan(alloc)
    # Place primaries first (round-robin), then replicas, so primaries spread evenly.
    for s in range(num_shards):
        alloc[s].append(nodes[s % len(nodes)])
    _place_missing(alloc, list(nodes), 1 + replicas, plan)
    return alloc


def rebalance(current: Allocation, nodes: Sequence[str], replicas: int) -> RebalancePlan:
    """Adapt ``current`` to the live ``nodes`` with as few copy moves as possible."""
    nodes = list(nodes)
    alloc: Allocation = {}
    plan = RebalancePlan(alloc)
    for shard, copies in current.items():
        alive = [n for n in copies if n in nodes]
        plan.removed += [(shard, n) for n in copies if n not in nodes]
        if alive and copies and alive[0] != copies[0]:
            plan.promoted.append((shard, alive[0]))
        alloc[shard] = alive
    _place_missing(alloc, nodes, 1 + replicas, plan)

    # Move copies from overloaded to underloaded nodes (e.g. after a node joins).
    while True:
        loads = _loads(alloc, nodes)
        hi = max(nodes, key=lambda n: (loads[n], -nodes.index(n)))
        lo = min(nodes, key=lambda n: (loads[n], nodes.index(n)))
        if loads[hi] - loads[lo] <= 1:
            break
        movable = [s for s in sorted(alloc) if hi in alloc[s] and lo not in alloc[s]]
        if not movable:
            break
        # Prefer moving a replica rather than a primary.
        shard = next((s for s in movable if alloc[s][0] != hi), movable[0])
        pos = alloc[shard].index(hi)
        alloc[shard][pos] = lo
        plan.removed.append((shard, hi))
        plan.added.append((shard, lo))
    return plan


def validate(alloc: Allocation) -> None:
    for shard, copies in alloc.items():
        if len(copies) != len(set(copies)):
            raise AssertionError(f"shard {shard} has two copies on one node: {copies}")
