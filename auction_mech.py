from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Bid:
    agent_id: str
    task_id: str
    value: float


class AuctionMechanism:
    """
    Simple per-task sealed-bid auction.
    Highest bid wins (first-price). Optionally can be extended to Vickrey.
    """

    def __init__(self, vickrey: bool = False):
        self.vickrey = vickrey
        self._bids_by_task: DefaultDict[str, List[Bid]] = defaultdict(list)
        self.allocated_tasks: set[str] = set()  # task_id -> winning agent_id (or None)

    def submit_bid(self, bid: Bid) -> None:
        if bid.task_id in self.allocated_tasks:
            raise ValueError(f"Task {bid.task_id} already allocated, cannot accept new bids.")
        self._bids_by_task[bid.task_id].append(bid)

    def clear_bids(self) -> None:
        self._bids_by_task.clear()

    def run_auction(self) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
        """
        Run independent auctions for each task.
        Returns:
          allocations: task_id -> winning agent_id (or None)
          payments: task_id -> amount the winner pays
        """
        allocations: Dict[str, Optional[str]] = {}
        payments: Dict[str, float] = {}

        for task_id, bids in self._bids_by_task.items():
            if not bids:
                allocations[task_id] = None
                payments[task_id] = 0.0
                continue

            sorted_bids = sorted(bids, key=lambda b: b.value, reverse=True)
            winner = sorted_bids[0]
            allocations[task_id] = winner.agent_id
            self.allocated_tasks.add(task_id)

            if self.vickrey:
                second_price = sorted_bids[1].value if len(sorted_bids) > 1 else 0.0
                payments[task_id] = second_price
            else:
                payments[task_id] = winner.value
            self.allocated_tasks.add(task_id)

        return allocations, payments
