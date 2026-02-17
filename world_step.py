import numpy as np
import random
from world_nearest_target import nearest_target
from auction_mech import AuctionMechanism, Bid
from settings import TARGET_SPEED, HUNTER_SPEED, WIDTH, HEIGHT, SEPARATION_DISTANCE


def world_step(world):

    # ===== TARGETS MOVE SMOOTHLY =====
    for t in world.targets:

        # small random steering change
        t.vel += np.random.uniform(-0.2, 0.2, 2)

        speed = np.linalg.norm(t.vel)
        if speed > TARGET_SPEED:
            t.vel = (t.vel / speed) * TARGET_SPEED

        # call entity move (assumes Entity.move delegates to entity_move)
        t.move()

    # ===== HUNTERS: build bids, run auction, then chase assigned targets =====

    # build sealed bids based on distance (higher value for closer targets)
    auction = AuctionMechanism(vickrey=False)

    # use index strings as agent/task ids (entities don't have persistent ids)
    for hi, h in enumerate(world.hunters):
        for ti, t in enumerate(world.targets):
            dist = np.linalg.norm(t.pos - h.pos)
            # value: closer targets are more valuable; clamp at 0
            value = max(0.0, 200.0 - dist)
            auction.submit_bid(Bid(agent_id=str(hi), task_id=str(ti), value=value))

    allocations, payments = auction.run_auction()

    # map agent index -> assigned target entity (or None)
    agent_target = {str(i): None for i in range(len(world.hunters))}
    for task_id, agent_id in allocations.items():
        if agent_id is not None:
            agent_target[agent_id] = int(task_id)

    for hi, h in enumerate(world.hunters):
        assigned_ti = agent_target.get(str(hi))
        if assigned_ti is not None:
            target = world.targets[assigned_ti]
        else:
            target = nearest_target(world, h)

        pursue = target.pos - h.pos
        dist = np.linalg.norm(pursue)

        if dist > 0:
            pursue = pursue / dist

        direction = pursue
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        h.vel = direction * HUNTER_SPEED
        h.move()

    # ===== LOCAL COLLISION-RESOLUTION TO MAINTAIN MINIMUM SEPARATION =====
    # push overlapping hunters apart so they don't join together
    n = len(world.hunters)
    for i in range(n):
        for j in range(i + 1, n):
            a = world.hunters[i]
            b = world.hunters[j]
            delta = a.pos - b.pos
            dist = np.linalg.norm(delta)
            min_sep = SEPARATION_DISTANCE
            if dist == 0:
                # identical positions: jitter them
                jitter = np.random.uniform(-0.5, 0.5, 2)
                a.pos += jitter
                b.pos -= jitter
                continue
            if dist < min_sep:
                overlap = min_sep - dist
                # move each by half the overlap along the separation vector
                push = (delta / dist) * (overlap / 2.0)
                a.pos += push
                b.pos -= push

    # ===== CAPTURE =====
    for h in world.hunters:
        for t in world.targets:

            dist = np.linalg.norm(h.pos - t.pos)

            if dist < 15:
                t.pos = np.array(
                    [
                        random.randint(50, WIDTH - 50),
                        random.randint(50, HEIGHT - 50),
                    ],
                    dtype=np.float32,
                )
                t.vel = np.random.uniform(-1, 1, 2)
