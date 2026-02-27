import numpy as np
import random


class UUV:
    def __init__(self, width, height):
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)


class Target:
    def __init__(self, width, height, drift_range=0.3):
        self.width = width
        self.height = height
        self.drift_range = float(drift_range)
        self.pos = np.array([
            random.uniform(0, width),
            random.uniform(0, height)
        ], dtype=np.float32)
        self.vel = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def move(self):
        drift = np.random.uniform(-self.drift_range, self.drift_range, size=2)
        self.pos += drift
        self.pos = np.clip(self.pos, [0, 0], [self.width, self.height])


class MultiAgentEnv:
    def __init__(
        self,
        width=900,
        height=600,
        n_uuv=3,
        n_targets=2,
        catch_radius=50,
        uuv_speed=8.0,
        target_drift=0.3,
        respawn_on_capture=False,
        world_mode=False,
        # World-mode params (matching `classical-sim`)
        spawn_interval=40,
        max_active_targets=15,
        min_active_targets=3,
        spawn_margin=50,
        target_speed=2.5,
        target_min_speed=1.2,
        target_burst_prob=0.08,
        separation_distance=80.0,
        separation_weight=2.0,
        ocean_flow_strength=1.5,
        ocean_flow_frequency=0.01,
        ocean_flow_scale=200.0,
        ocean_vortex_enabled=True,
    ):
        self.width = width
        self.height = height
        self.n_uuv = n_uuv
        self.n_targets = n_targets
        self.catch_radius = float(catch_radius)
        self.uuv_speed = float(uuv_speed)
        self.target_drift = float(target_drift)
        self.respawn_on_capture = bool(respawn_on_capture)

        self.world_mode = bool(world_mode)
        self.spawn_interval = int(spawn_interval)
        self.max_active_targets = int(max_active_targets)
        self.min_active_targets = int(min_active_targets)
        self.spawn_margin = int(spawn_margin)
        self.target_speed = float(target_speed)
        self.target_min_speed = float(target_min_speed)
        self.target_burst_prob = float(target_burst_prob)
        self.separation_distance = float(separation_distance)
        self.separation_weight = float(separation_weight)
        self.ocean_flow_strength = float(ocean_flow_strength)
        self.ocean_flow_frequency = float(ocean_flow_frequency)
        self.ocean_flow_scale = float(ocean_flow_scale)
        self.ocean_vortex_enabled = bool(ocean_vortex_enabled)

        self.time_step = 0
        self.spawn_counter = 0
        self.captures = 0
        self.reset()

    def _spawn_targets_world(self):
        spawn_count = random.randint(3, 5)
        for _ in range(spawn_count):
            if len(self.targets) >= self.max_active_targets:
                break
            t = Target(self.width, self.height, drift_range=0.0)
            t.pos = np.array(
                [
                    random.uniform(self.spawn_margin, self.width - self.spawn_margin),
                    random.uniform(self.spawn_margin, self.height - self.spawn_margin),
                ],
                dtype=np.float32,
            )
            # Initialize with a meaningful speed so targets visibly move.
            angle = random.uniform(0, 2 * np.pi)
            base_speed = random.uniform(0.6 * self.target_speed, 1.2 * self.target_speed)
            t.vel = np.array([np.cos(angle) * base_speed, np.sin(angle) * base_speed], dtype=np.float32)
            self.targets.append(t)

    def _get_ocean_flow(self, x, y):
        t = self.time_step * self.ocean_flow_frequency
        flow_x = (
            np.sin(x / self.ocean_flow_scale + t) * 0.5
            + np.sin(y / self.ocean_flow_scale * 0.5) * 0.3
        )
        flow_y = (
            np.cos(y / self.ocean_flow_scale + t) * 0.5
            + np.cos(x / self.ocean_flow_scale * 0.5) * 0.3
        )

        if self.ocean_vortex_enabled:
            center_x = self.width / 2
            center_y = self.height / 2
            dx = x - center_x
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2) + 1.0
            vortex_x = -dy / (dist**2) * 0.3
            vortex_y = dx / (dist**2) * 0.3
            flow_x += vortex_x
            flow_y += vortex_y

        return np.array([flow_x, flow_y], dtype=np.float32) * self.ocean_flow_strength

    def _separation_force(self, uuv):
        force = np.zeros(2, dtype=np.float32)
        for other in self.uuvs:
            if other is uuv:
                continue
            d = other.pos - uuv.pos
            dist = float(np.linalg.norm(d))
            if 0.0 < dist < self.separation_distance:
                force += (-d / dist) * (1.0 / dist)
        return force

    def _bounce_in_bounds(self, obj):
        if obj.pos[0] <= 0 or obj.pos[0] >= self.width:
            obj.vel[0] *= -1
        if obj.pos[1] <= 0 or obj.pos[1] >= self.height:
            obj.vel[1] *= -1
        obj.pos[0] = float(np.clip(obj.pos[0], 0, self.width))
        obj.pos[1] = float(np.clip(obj.pos[1], 0, self.height))

    def reset(self):
        self.uuvs = [UUV(self.width, self.height) for _ in range(self.n_uuv)]
        self.time_step = 0
        self.spawn_counter = 0
        if self.world_mode:
            self.targets = []
            # Seed some targets immediately
            self._spawn_targets_world()
            while len(self.targets) < self.min_active_targets:
                self._spawn_targets_world()
        else:
            self.targets = [Target(self.width, self.height, drift_range=self.target_drift) for _ in range(self.n_targets)]

        # FIXED ASSIGNMENT PER EPISODE
        self.assignments = self.assign_targets()

        return self._get_obs()

    def assign_targets(self):
        assignments = {}
        taken = set()

        for i, uuv in enumerate(self.uuvs):
            distances = []
            for j, target in enumerate(self.targets):
                if j in taken:
                    distances.append(np.inf)
                else:
                    distances.append(np.linalg.norm(uuv.pos - target.pos))

            chosen = int(np.argmin(distances)) if len(distances) else 0
            assignments[i] = chosen
            taken.add(chosen)

        return assignments

    def step(self, actions):

        rewards = []
        done = False

        # In "final product" world-mode, targets move and the best assignment changes constantly.
        if self.world_mode:
            self.assignments = self.assign_targets()

        if self.world_mode:
            self.time_step += 1
            self.spawn_counter += 1

            if self.spawn_counter >= self.spawn_interval:
                self.spawn_counter = 0
                self._spawn_targets_world()

            while len(self.targets) < self.min_active_targets:
                self._spawn_targets_world()

            # Update targets (velocity + bursts + ocean flow)
            for t in self.targets:
                if random.random() < self.target_burst_prob:
                    angle = random.uniform(0, 2 * np.pi)
                    burst_speed = random.uniform(1.5, self.target_speed * 1.8)
                    t.vel = np.array([np.cos(angle) * burst_speed, np.sin(angle) * burst_speed], dtype=np.float32)

                flow = self._get_ocean_flow(float(t.pos[0]), float(t.pos[1]))
                t.vel = t.vel + flow * 0.2

                speed = float(np.linalg.norm(t.vel))
                if speed < self.target_min_speed:
                    angle = random.uniform(0, 2 * np.pi)
                    t.vel = t.vel + np.array(
                        [np.cos(angle) * self.target_min_speed, np.sin(angle) * self.target_min_speed],
                        dtype=np.float32,
                    )
                    speed = float(np.linalg.norm(t.vel))
                if speed > self.target_speed * 1.8 and speed > 0:
                    t.vel = (t.vel / speed) * (self.target_speed * 1.8)

                t.pos = t.pos + t.vel
                self._bounce_in_bounds(t)
        else:
            # Move targets (simple drift used for training)
            for t in self.targets:
                t.move()

        to_remove = []
        for i, uuv in enumerate(self.uuvs):

            a = np.asarray(actions[i], dtype=np.float32)
            if self.world_mode:
                sep = self._separation_force(uuv)
                direction = a + sep * self.separation_weight
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction = direction / norm
                else:
                    direction = np.zeros(2, dtype=np.float32)
                uuv.vel = direction * self.uuv_speed
                uuv.pos = uuv.pos + uuv.vel
                self._bounce_in_bounds(uuv)
            else:
                uuv.pos += a * self.uuv_speed
                uuv.pos = np.clip(uuv.pos, [0, 0], [self.width, self.height])

            if not self.targets:
                rewards.append(-0.005)
                continue

            target = self.targets[self.assignments.get(i, 0) % len(self.targets)]
            dist = np.linalg.norm(uuv.pos - target.pos)

            # Better reward shaping - encourages consistent progress
            max_distance = np.sqrt(self.width**2 + self.height**2)
            normalized_dist = np.clip(dist / max_distance, 0, 1)
            distance_reward = (1.0 - normalized_dist) * 1.0  # Scaled to 1.0 max per step

            # Minimal time penalty
            time_penalty = -0.005

            reward = distance_reward + time_penalty

            # Success bonus for capturing target
            if dist < self.catch_radius:
                reward += 5.0  # Meaningful success bonus
                if self.world_mode:
                    self.captures += 1
                    if self.respawn_on_capture:
                        target.pos = np.array(
                            [
                                random.uniform(self.spawn_margin, self.width - self.spawn_margin),
                                random.uniform(self.spawn_margin, self.height - self.spawn_margin),
                            ],
                            dtype=np.float32,
                        )
                        angle = random.uniform(0, 2 * np.pi)
                        base_speed = random.uniform(0.6 * self.target_speed, 1.2 * self.target_speed)
                        target.vel = np.array(
                            [np.cos(angle) * base_speed, np.sin(angle) * base_speed],
                            dtype=np.float32,
                        )
                    else:
                        # Remove captured target and keep the show going
                        to_remove.append(target)
                elif self.respawn_on_capture:
                    self.captures += 1
                    target.pos = np.array(
                        [
                            random.uniform(0, self.width),
                            random.uniform(0, self.height),
                        ],
                        dtype=np.float32,
                    )
                else:
                    done = True

            rewards.append(reward)

        if self.world_mode and to_remove:
            for t in to_remove:
                try:
                    self.targets.remove(t)
                except ValueError:
                    pass

            while len(self.targets) < self.min_active_targets:
                self._spawn_targets_world()

        return self._get_obs(), rewards, done

    def _get_obs(self):

        obs = []

        target_positions = []
        for t in self.targets:
            target_positions.extend(
                t.pos / np.array([self.width, self.height])
            )

        for uuv in self.uuvs:
            uuv_pos = uuv.pos / np.array([self.width, self.height])
            state = np.concatenate([uuv_pos, target_positions])
            obs.append(state)

        return obs