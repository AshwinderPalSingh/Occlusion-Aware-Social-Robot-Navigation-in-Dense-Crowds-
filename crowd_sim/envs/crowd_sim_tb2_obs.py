import gym
import numpy as np
from numpy.linalg import norm
import os
import copy
from collections import deque
import matplotlib.pyplot as plt
try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

from crowd_sim.envs.crowd_sim_tb2 import CrowdSim3DTB
from crowd_sim.envs.occlusion_grid import build_grids

'''
Everything is the same as CrowdSimVarNum, except the obstacle vertices are in observation space
'''

# This class is the same as CrowdSim3DTB except observation space
class CrowdSim3DTbObs(CrowdSim3DTB):
    def __init__(self):
        super().__init__()
        # number of steps before we consider a human is stuck
        self.human_timeout_len = 10
        # Stage 3: Acoustic cooldown timer
        self.acoustic_cooldown_timer = 0
        self.robot_is_speaking = False
        # Path B metrics: ids of occluded humans localized by audio this episode. Kept as
        # a set of ids (not a running sum) because a human vocalizes for return_duration
        # consecutive steps -- summing per-step counts would multiply-count one detection.
        self._occluded_heard_ids = set()
        self._num_occluded_heard = 0
        self._num_occluded_newly_heard = 0
        self._last_episode_occluded_heard = 0

    def set_observation_space(self):
        d={}
        # Stage 3: robot_node gets a +1 speak-flag feature when acoustic is enabled
        # (appended in generate_ob() below) -- this must stay in sync with that, or the
        # declared gym space disagrees with what generate_ob() actually returns.
        acoustic_dim = 1 if (hasattr(self.config, 'acoustic') and self.config.acoustic.enabled) else 0
        # we use 'absolute' here
        if self.config.ob_space.robot_state == 'absolute':
            # robot px, py (in world frame), and theta (heading angle in z axis)
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5 + acoustic_dim,), dtype=np.float32)
        else:
            # gx-px, gy-py, theta
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3 + acoustic_dim,), dtype = np.float32)
        # robot vx, vy (in world frame)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # make sure there's at least one human
        # add_human_vel is True
        pred_len = self.config.sim.predict_steps * 2 if hasattr(self.config.sim, 'predict_method') and self.config.sim.predict_method != 'none' else 0
        # +1 trailing confidence column when enabled (see generate_ob)
        pred_len += 1 if (hasattr(self.config, 'acoustic') and self.config.acoustic.enabled
                          and getattr(self.config.acoustic, 'confidence_channel', False)) else 0
        if self.config.ob_space.add_human_vel:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 4 + pred_len),
                                                dtype=np.float32)
        else:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 2 + pred_len),
                                                dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        # obstacle representations methods:
        # Relative coordinates of 4 vertices w.r.t. the robot
        # # [lower left, lower_right, upper_right, upper_left]
        # where lower left = [lower left x  - robot.px, lower left y - robot.py], and same for others
        d['obstacle_vertices'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_obs_num), 8,), dtype=np.float32)

        # number of obstacles
        d['obstacle_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # 3. raw lidar point cloud from robot's 2D lidar
        d['point_clouds'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

        # Stage 2: occlusion-inference grids (only used when sim.occlusion_method == 'vae')
        if hasattr(self.config.sim, 'occlusion_method') and self.config.sim.occlusion_method == 'vae':
            grid_size = self.config.pas.grid_size
            d['grid'] = gym.spaces.Box(low=0, high=1, shape=(1, grid_size, grid_size), dtype=np.float32)
            d['label_grid'] = gym.spaces.Box(low=0, high=1, shape=(1, grid_size, grid_size), dtype=np.float32)

        # Stage 3: Acoustic edges — per-human response state visible to the policy
        if hasattr(self.config, 'acoustic') and self.config.acoustic.enabled:
            # For each human: [is_responding (0/1), signal_strength, relative_px, relative_py]
            d['acoustic_edges'] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max(1, self.max_human_num), 4),
                dtype=np.float32
            )
            # Path B: [is_heard, noisy_rel_px, noisy_rel_py, localization_quality]
            if getattr(self.config.acoustic, 'return_enabled', False):
                d['acoustic_returns'] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max(1, self.max_human_num), 4),
                    dtype=np.float32
                )

        self.observation_space=gym.spaces.Dict(d)


    def generate_circle_crossing_human(self, static=False):
        human = super().generate_circle_crossing_human(static=static)

        # use a deque to store the recent speed of each human
        human.recent_speeds = deque(maxlen=self.human_timeout_len*2)
        human.recent_speeds.append(norm([human.vx, human.vy]))

        # Stage 3: Initialize acoustic response state for this human
        if hasattr(self.config, 'acoustic') and self.config.acoustic.enabled:
            human.acoustic_responding = False
            human.acoustic_pending = False
            human.acoustic_delay_timer = 0
            human.acoustic_response_timer = 0
            human.acoustic_returning = False
            human.acoustic_return_timer = 0
            human.acoustic_belief_world = None
            human.acoustic_belief_timer = 0
            human.acoustic_belief_quality = 0.0
            human.acoustic_force_dir = np.array([0., 0.])
            human.acoustic_force_strength = 0.0
            human.acoustic_signal_strength = 0.0
            human.acoustic_compliant = (np.random.random() < self.config.acoustic.compliance_rate)
            human.original_gx = None
            human.original_gy = None

        return human

    def step(self, action, update=True):
        # Stage 3: Extract speak flag from discrete action before parent step
        self.robot_is_speaking = False
        acoustic_enabled = hasattr(self.config, 'acoustic') and self.config.acoustic.enabled
        num_newly_triggered = 0
        num_newly_triggered_occluded = 0

        if acoustic_enabled and self.config.env.action_space == 'discrete':
            # Unwrap from numpy array if needed
            act_idx = action[0] if isinstance(action, np.ndarray) else action
            if act_idx >= 9:
                # This is a speak action — check cooldown
                if self.acoustic_cooldown_timer <= 0:
                    self.robot_is_speaking = True
                    self.acoustic_cooldown_timer = self.config.acoustic.cooldown
                # Convert action index back to movement-only (0-8)
                action = np.array([act_idx - 9]) if isinstance(action, np.ndarray) else act_idx - 9

            # Decrement cooldown each step
            if self.acoustic_cooldown_timer > 0:
                self.acoustic_cooldown_timer -= 1

        # Apply acoustic force to humans BEFORE they compute ORCA actions in parent step
        if self.robot_is_speaking:
            num_newly_triggered, num_newly_triggered_occluded = self._apply_acoustic_force()

        # Update acoustic response timers for all humans (even if not speaking this step)
        if acoustic_enabled:
            self._update_acoustic_responses()

        # Call parent step (computes human ORCA, physics, reward, etc.)
        ob, reward, done, info = super().step(action, update=update)

        for human in self.humans:
            human.recent_speeds.append(norm([human.vx, human.vy]))
        # For every self.human_timeout_len steps, check if a human is stuck for a while
        # if so, change its goal
        # Don't do this in csl_workspace scenarios because human routes are designed
        if self.config.env.scenario == 'circle_crossing':
            if self.step_counter % self.human_timeout_len == 0:
                for i in range(len(self.humans)):
                    if sum(self.humans[i].recent_speeds) / len(self.humans[i].recent_speeds) < 0.1:
                        self.update_human_goal(self.humans[i])

        # Stage 3: acoustic reward shaping
        if self.robot_is_speaking and acoustic_enabled:
            reward += self.config.acoustic.speak_penalty
            # Path B info-gain bonus: paid ONLY for newly triggering a human the robot
            # currently CANNOT SEE. Those are the ones whose vocal reply will actually
            # tell it something it did not already know -- beeping at humans already in
            # plain view yields no information and so earns nothing here, leaving only
            # the speak_penalty. Restricting to *newly* triggered humans (an already
            # responding/pending human can't re-trigger) also means re-beeping the same
            # hidden crowd during their response window cannot re-farm this bonus.
            reward += self.config.acoustic.info_gain_coef * num_newly_triggered_occluded

        # Reset acoustic state on episode done. NOTE: the Path B detection tally
        # (_occluded_heard_ids / _num_occluded_newly_heard) is deliberately NOT cleared
        # here -- evaluation.py reads it after done to record the episode's total. It is
        # cleared at episode START instead, in _init_acoustic_state().
        if done and acoustic_enabled:
            self.acoustic_cooldown_timer = 0
            self.robot_is_speaking = False
            # Latch this episode's detection tally NOW. evaluation.py reads it after the
            # episode loop exits, but by then test.py has already called reset() for the
            # next episode, which runs _init_acoustic_state() and zeroes the live
            # counter -- so reading the live counter always returned 0.
            self._last_episode_occluded_heard = self._num_occluded_newly_heard

        return ob, reward, done, info

    def _apply_acoustic_force(self):
        """Stage 3: trigger acoustic response for compliant, in-range humans.

        This only starts each human's response_delay countdown and rolls their
        (distance-decayed) compliance once per beep; the actual repulsive goal-shift is
        applied later, once the delay elapses (see _update_acoustic_responses), so the
        robot_node's binary speak flag and the eventual behavior change are correctly
        separated by response_delay steps instead of both happening on this exact step.

        Returns the number of humans newly triggered this call (used for the info-gain
        reward in step()) -- deliberately excludes humans already acoustic_responding or
        acoustic_pending, so this can't be farmed by re-beeping the same crowd.
        """
        cfg = self.config.acoustic
        robot_pos = np.array([self.robot.px, self.robot.py])
        num_newly_triggered = 0
        num_newly_triggered_occluded = 0

        for idx, human in enumerate(self.humans):
            if human.isObstacle or human.acoustic_responding or human.acoustic_pending:
                continue  # one beep, one response cycle per human

            human_pos = np.array([human.px, human.py])
            signal = self.compute_acoustic_strength(robot_pos, human_pos)

            if signal <= 0.01 or not human.acoustic_compliant:
                continue

            # Distance-decayed compliance (L2B-style), rolled once per beep -- not
            # re-rolled every step, since the human is now either pending or responding.
            comply_prob = cfg.compliance_rate * signal
            if np.random.random() >= comply_prob:
                continue

            dist = norm(human_pos - robot_pos)
            direction = (human_pos - robot_pos) / dist if dist > 0.01 else np.array([1.0, 0.0])

            human.acoustic_pending = True
            human.acoustic_delay_timer = cfg.response_delay
            human.acoustic_force_dir = direction
            human.acoustic_signal_strength = signal
            human.acoustic_force_strength = cfg.alpha * signal
            num_newly_triggered += 1
            # A human the robot currently cannot see is one it is about to LEARN about
            # (they will vocalize a reply in response_delay steps, and that reply carries
            # around the obstacle). This is the information-gain event the reward targets.
            if idx < len(self.human_visibility) and not self.human_visibility[idx]:
                num_newly_triggered_occluded += 1

        return num_newly_triggered, num_newly_triggered_occluded

    def _build_acoustic_returns(self):
        """Path B: what the robot HEARS from humans vocalizing a reply to its beep.

        This is the sensing half of "social echolocation", and the only channel in the
        system that can carry information about a human the camera cannot see: audio
        diffracts around corners, vision does not. Gated by audible range, NOT by
        line-of-sight -- that is the entire point.

        It is deliberately a poor sensor rather than a wallhack:
          - measurements carry bearing and range noise (audio localization is coarse),
          - noise is further multiplied when the speaker is occluded, since diffraction
            around an obstacle destroys the interaural cues used to localize,
          - it is transient: audible only during the human's short return_duration
            window, so the robot must fuse it into its PaS-VAE belief rather than
            reading positions off it continuously.

        Returns (returns_array, num_occluded_heard) where returns_array is
        [max_human_num, 4] = [is_heard, noisy_rel_px, noisy_rel_py, localization_quality],
        indexed by human id (caller reindexes it to match spatial_edges slot order).
        """
        cfg = self.config.acoustic
        returns = np.zeros((max(1, self.max_human_num), 4))
        num_occluded_heard = 0

        if not getattr(cfg, 'return_enabled', False):
            return returns, 0

        robot_pos = np.array([self.robot.px, self.robot.py])

        belief_dur = float(getattr(cfg, 'belief_duration', 0) or 0)

        for i in range(self.human_num):
            human = self.humans[i]

            # (a) A fresh fix, if this human is vocalizing in range right now. The noisy
            #     estimate is stored in WORLD coordinates so it stays meaningful as the
            #     robot moves; drawing the noise once (here, not every step) keeps the
            #     remembered belief self-consistent instead of jittering each frame.
            if getattr(human, 'acoustic_returning', False):
                true_vec = np.array([human.px, human.py]) - robot_pos
                true_range = norm(true_vec)
                if 1e-6 < true_range <= cfg.return_max_range:
                    noise_mult = cfg.return_occluded_noise_mult if not self.human_visibility[i] else 1.0
                    meas_bearing = (np.arctan2(true_vec[1], true_vec[0])
                                    + np.random.normal(0., cfg.return_bearing_noise * noise_mult))
                    meas_range = max(0.1, true_range
                                     + np.random.normal(0., cfg.return_range_noise * noise_mult))
                    _fix = robot_pos + meas_range * np.array(
                        [np.cos(meas_bearing), np.sin(meas_bearing)])
                    # Average the independent fixes gathered across one utterance instead
                    # of keeping only the newest. A reply is audible for return_duration
                    # steps, so this is ~sqrt(n) less localization error, and it matches
                    # the physics: a longer utterance localizes better than a blip.
                    # acoustic_belief_n is zeroed when a NEW utterance starts, so a later
                    # reply is never averaged together with a stale earlier one.
                    _n = getattr(human, 'acoustic_belief_n', 0) + 1
                    if _n == 1 or human.acoustic_belief_world is None:
                        human.acoustic_belief_world = _fix
                    else:
                        human.acoustic_belief_world = (
                            human.acoustic_belief_world + (_fix - human.acoustic_belief_world) / _n)
                    human.acoustic_belief_n = _n
                    human.acoustic_belief_timer = belief_dur if belief_dur > 0 else 1
                    # Store the fix's expected POSITION ERROR (metres), not an abstract
                    # score. Averaging n independent fixes shrinks it by sqrt(n). This is
                    # what confidence is derived from below -- an earlier version scored
                    # confidence as (1/noise_mult)*sqrt(n), which climbed to 0.89 for a
                    # fresh occluded fix and was therefore nearly indistinguishable from a
                    # confirmed visual detection at 1.0, despite ~0.7m of real error.
                    human.acoustic_belief_err = (cfg.return_range_noise * noise_mult) / np.sqrt(_n)

            # (b) Emit the belief while it is still live -- this is what actually reaches
            #     the policy, and it outlives the voice by belief_duration steps.
            if getattr(human, 'acoustic_belief_timer', 0) <= 0:
                continue
            if getattr(human, 'acoustic_belief_world', None) is None:
                continue

            rel = human.acoustic_belief_world - robot_pos
            rel_px, rel_py = float(rel[0]), float(rel[1])
            if self.config.ob_space.human_state_frame == 'robot':
                rel_px, rel_py = self.world_to_robot([rel_px, rel_py])

            # Confidence = how the estimate's expected error compares to the physical
            # collision envelope. The belief has no motion model, so error grows by the
            # distance a human could have walked since the fix. Expressed this way the
            # number is physically meaningful (0.5 => error equals the envelope) and
            # stays clearly separated from a confirmed visual detection's 1.0.
            _age_steps = max(0.0, belief_dur - human.acoustic_belief_timer) if belief_dur > 0 else 0.0
            _drift = self.config.humans.v_pref * _age_steps * self.config.env.time_step
            _err = getattr(human, 'acoustic_belief_err', cfg.return_range_noise) + _drift
            _envelope = self.config.humans.radius + self.config.robot.radius
            quality = _envelope / (_envelope + _err)

            returns[human.id] = [1.0, rel_px, rel_py, quality]
            if not self.human_visibility[i]:
                num_occluded_heard += 1
                if human.id not in self._occluded_heard_ids:
                    self._occluded_heard_ids.add(human.id)
                    self._num_occluded_newly_heard += 1

        return returns, num_occluded_heard

    def _update_acoustic_responses(self):
        """Stage 3: advance each human's delay -> response -> smooth-return timers."""
        cfg = self.config.acoustic
        for human in self.humans:
            # Path B: the vocal reply is audible for its own short window, ticked
            # independently of the (much longer) movement-response state machine below.
            if human.acoustic_returning:
                human.acoustic_return_timer -= 1
                if human.acoustic_return_timer <= 0:
                    human.acoustic_returning = False

            # Path B: the remembered audio fix ages out independently of the voice.
            if getattr(human, 'acoustic_belief_timer', 0) > 0:
                human.acoustic_belief_timer -= 1
                if human.acoustic_belief_timer <= 0:
                    human.acoustic_belief_world = None
                    human.acoustic_belief_quality = 0.0

            if human.acoustic_pending:
                human.acoustic_delay_timer -= 1
                if human.acoustic_delay_timer > 0:
                    continue
                # Delay elapsed -- start the actual repulsive response now.
                human.acoustic_pending = False
                human.acoustic_responding = True
                human.acoustic_response_timer = cfg.response_duration

                # Path B: the human also vocalizes a reply as they step aside. This is
                # what the robot actually *hears* (see _build_acoustic_returns) -- the
                # sensing half of the mechanism, as opposed to the path-clearing half.
                if getattr(cfg, 'return_enabled', False):
                    human.acoustic_returning = True
                    human.acoustic_return_timer = cfg.return_duration
                    human.acoustic_belief_n = 0  # start a fresh averaging window

                if human.original_gx is None:
                    human.original_gx = human.gx
                    human.original_gy = human.gy

                signal = getattr(human, 'acoustic_signal_strength', 1.0)
                human.gx = human.px + human.acoustic_force_dir[0] * cfg.goal_shift_scale * signal
                human.gy = human.py + human.acoustic_force_dir[1] * cfg.goal_shift_scale * signal
                continue

            if not human.acoustic_responding:
                continue

            human.acoustic_response_timer -= 1

            # Response expired → smooth return phase
            if human.acoustic_response_timer <= 0:
                if human.original_gx is not None:
                    # Smoothly interpolate back to original goal
                    t = 1.0 / max(1, cfg.return_smoothing)
                    human.gx = human.gx + t * (human.original_gx - human.gx)
                    human.gy = human.gy + t * (human.original_gy - human.gy)

                    # Close enough to original goal → snap back and reset
                    if abs(human.gx - human.original_gx) < 0.05 and abs(human.gy - human.original_gy) < 0.05:
                        human.gx = human.original_gx
                        human.gy = human.original_gy
                        human.original_gx = None
                        human.original_gy = None
                        human.acoustic_responding = False
                        human.acoustic_force_strength = 0.0

    def ray_test_no_humans(self):
        """
        perform a lidar ray test on all obstacles in the scene WITHOUT humans
        save the current range readings in self.closest_hit_dist
        """
        # relocate all pybullet human cylinders to somewhere out of lidar range
        for i in range(self.human_num):
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [30, 30, self.config.humans.height/2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))
        # lidar ray test
        self.ray_test()

        # relocate humans back based on their original positions
        for i in range(self.human_num):
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [self.humans[i].px, self.humans[i].py, self.config.humans.height/2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))

    # rotate a velocity vector [x, y] from world frame to robot frame
    def world_to_robot(self, vec):
        x, y = vec
        # rotation matrix from world to robot frame
        rot_angle = -(self.robot.theta - np.pi/2)
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle), np.cos(rot_angle)]])
        vec_trans = np.matmul(R, np.array([[x], [y]]))
        return np.array([vec_trans[0, 0], vec_trans[1, 0]])

    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        if self.config.ob_space.robot_state == 'absolute':
            ob['robot_node'] = self.robot.get_changing_state_list()
        else:
            ob['robot_node'] = self.robot.get_changing_state_list_goal_offset()

        # Stage 3: append binary speak flag -- matches the +1 declared in
        # set_observation_space(). Uses self.robot_is_speaking (set in step(), from the
        # action BEFORE it's remapped down to the 9-action space) rather than the old
        # self._spoke_this_step, which nothing in this class's step() ever set.
        if hasattr(self.config, 'acoustic') and self.config.acoustic.enabled:
            spoke = 1.0 if getattr(self, 'robot_is_speaking', False) else 0.0
            ob['robot_node'] = list(ob['robot_node']) + [spoke]

        self.update_last_human_states(self.human_visibility, reset=reset)

        # Stage 2: occlusion grids must be built from the camera-FOV ray test that
        # get_num_human_in_fov() just ran (self.closest_hit_dist/closest_hit_id, indexed
        # by cam_ray_angles) -- ray_test()/ray_test_no_humans() further below overwrites
        # those same attributes with the full 360 lidar sweep (indexed by ray_angles).
        if hasattr(self.config.sim, 'occlusion_method') and self.config.sim.occlusion_method == 'vae':
            sensor_grid, label_grid = build_grids(self)
            ob['grid'] = sensor_grid
            ob['label_grid'] = label_grid

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # ([relative px, relative py, disp_x, disp_y, future_px, future_py...], human id)
        # make sure there's at least one placeholder
        pred_len = 0
        if hasattr(self.config.sim, 'predict_method') and self.config.sim.predict_method != 'none':
            pred_len = self.config.sim.predict_steps * 2
        # NOTE: the confidence column is concatenated after sorting, not allocated here.
            
        if self.config.ob_space.add_human_vel:
            all_spatial_edges = np.ones((max(1, self.max_human_num), 4 + pred_len)) * np.inf
        else:
            all_spatial_edges = np.ones((max(1, self.max_human_num), 2 + pred_len)) * np.inf
            
        # Calculate future positions if prediction is enabled
        if hasattr(self.config.sim, 'predict_method') and self.config.sim.predict_method != 'none':
            predicted_states = self.calc_human_future_traj(method=self.config.sim.predict_method)
            # [human_num, pred_steps, 2]
            predicted_pos = np.transpose(predicted_states[1:, :, :2], (1, 0, 2)) - np.array([self.robot.px, self.robot.py])
            # If using robot frame, transform predicted positions
            if self.config.ob_space.human_state_frame == 'robot':
                for h in range(predicted_pos.shape[0]):
                    for s in range(predicted_pos.shape[1]):
                        predicted_pos[h, s] = self.world_to_robot(predicted_pos[h, s])

        # robot.vx and vy are in world frame, transform to robot frame first
        v_robot_robFrame = self.world_to_robot([self.robot.vx, self.robot.vy])
        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot (in world frame)
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                # in self.last_human_states, the human relative positions are in world frame, transform to robot frame for sim2real
                if self.config.ob_space.human_state_frame == 'robot':
                    all_spatial_edges[self.humans[i].id, :2] = self.world_to_robot(relative_pos)
                else:
                    all_spatial_edges[self.humans[i].id, :2] = relative_pos
                    
                if not self.config.ob_space.add_human_vel and hasattr(self.config.sim, 'predict_method') and self.config.sim.predict_method != 'none':
                    all_spatial_edges[self.humans[i].id, 2:] = predicted_pos[i].reshape(-1)
                    
                if self.config.ob_space.add_human_vel:
                    if self.config.ob_space.human_state_frame == 'robot':
                        # in self.last_human_states, the human velocities are in world frame, transform to robot frame for sim2real
                        v_human = self.world_to_robot(self.last_human_states[i, 2:4])
                        # subtract robot velocity from human velocity, to get relative velocity of this human w.r.t. robot
                        if self.config.ob_space.human_vel == 'relative':
                            all_spatial_edges[self.humans[i].id, 2:4] = v_human - v_robot_robFrame
                        else:
                            all_spatial_edges[self.humans[i].id, 2:4] = v_human
                    else:
                        all_spatial_edges[self.humans[i].id, 2:4] = self.last_human_states[i, 2:4]
                        
                # append future predictions
                if hasattr(self.config.sim, 'predict_method') and self.config.sim.predict_method != 'none':
                    if self.config.ob_space.add_human_vel:
                        all_spatial_edges[self.humans[i].id, 4:] = predicted_pos[i].reshape(-1)
                    else:
                        all_spatial_edges[self.humans[i].id, 2:] = predicted_pos[i].reshape(-1)

        # ---- Path B: fold audio-localized humans into the graph as detected agents ----
        # Without this, an occluded-but-heard human keeps its [inf -> 15] sentinel row,
        # sorts to the very end, and lands beyond detected_human_num -- which is exactly
        # what create_attn_mask() masks off. The echolocation fix would be computed,
        # embedded, and then discarded by the attention layer before reaching the policy.
        # Giving them a real (noisy) position row and counting them as detected is what
        # actually lets the robot reason about people it cannot see.
        # Per-agent confidence, appended as the final spatial_edges column so the policy
        # can tell a confirmed sighting (1.0) from a decaying audio estimate (<1.0) from
        # an empty slot (0.0). Without this the network cannot learn to discount stale
        # beliefs -- the failure mode that made v2 worse than v1.
        _use_conf = (hasattr(self.config, 'acoustic') and self.config.acoustic.enabled
                     and getattr(self.config.acoustic, 'confidence_channel', False))
        _conf = np.zeros(max(1, self.max_human_num))
        for _i in range(self.human_num):
            if self.human_visibility[_i]:
                _conf[self.humans[_i].id] = 1.0   # confirmed this step

        num_audio_detected = 0
        if hasattr(self.config, 'acoustic') and self.config.acoustic.enabled:
            _ac_returns, self._num_occluded_heard = self._build_acoustic_returns()
            for _i in range(self.human_num):
                _hid = self.humans[_i].id
                if (not self.human_visibility[_i]) and _ac_returns[_hid, 0] > 0.5:
                    # audio fix gives position only -- velocity and future prediction are
                    # unknown from a single bearing/range measurement, so leave them zero.
                    # acoustic_returns' quality channel tells the network this row is
                    # audio-derived and coarser than a visual detection.
                    _apos = _ac_returns[_hid, 1:3]
                    all_spatial_edges[_hid, :] = 0.
                    all_spatial_edges[_hid, :2] = _apos
                    # Fill the prediction slots with the audio position (constant-position
                    # assumption). Leaving them at 0 would be actively WRONG, not neutral:
                    # these columns are RELATIVE to the robot, so 0 reads as "this human is
                    # predicted to sit exactly on the robot" at every future step.
                    # Velocity stays 0 -- a single bearing/range fix carries no velocity.
                    if pred_len > 0:
                        _ps = 4 if self.config.ob_space.add_human_vel else 2
                        _npred = self.config.sim.predict_steps * 2
                        all_spatial_edges[_hid, _ps:_ps + _npred] = np.tile(_apos, _npred // 2)
                    # confidence for this row = the belief's decayed quality
                    _conf[_hid] = float(_ac_returns[_hid, 3])
                    num_audio_detected += 1
        else:
            _ac_returns = None

        # sort all humans by distance (invisible humans will be in the end automatically).
        # Use argsort (not the plain sorted() this used to be) so the same permutation can
        # be reused below for acoustic_edges -- otherwise slot k in spatial_edges and slot k
        # in acoustic_edges would describe different humans (spatial_edges is distance-sorted,
        # acoustic_edges was id-indexed), and selfAttn_srnn_merge.py fuses them per-slot.
        sort_order = np.argsort(np.linalg.norm(all_spatial_edges[:, :2], axis=1))
        ob['spatial_edges'] = all_spatial_edges[sort_order]
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        if _use_conf:
            # appended last, after the inf->15 fill, so empty slots keep confidence 0.0
            # rather than being swept to the 15 sentinel like the geometry columns.
            ob['spatial_edges'] = np.concatenate(
                [ob['spatial_edges'], _conf[sort_order][:, None]], axis=1)

        # include audio-localized humans, else the attention mask drops them (see above)
        ob['detected_human_num'] = num_visibles + num_audio_detected
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # obstacle representations methods:
        # 1. relative coordinates of 4 vertices w.r.t. the robot

        ob['obstacle_vertices'] = np.ones((max(1, self.max_obs_num), 8,)) * 15
        # obstacle vertices in world frame
        if self.config.ob_space.human_state_frame == 'robot':
            cur_obs_vertices = np.array(self.obstacle_coord) - np.array([self.robot.px, self.robot.py])
            # convert obstacle vertics from world to robot frame
            cur_obs_vertices_rob_frame = np.zeros((self.obs_num, 4, 2))
            for i in range(self.obs_num):
                for j in range(4):
                    cur_obs_vertices_rob_frame[i, j] = self.world_to_robot(cur_obs_vertices[i, j])

            ob['obstacle_vertices'][:self.obs_num] = cur_obs_vertices_rob_frame.reshape(self.obs_num, -1)
        else:
            cur_obs_vertices = (np.array(self.obstacle_coord) - np.array([self.robot.px, self.robot.py])).reshape(self.obs_num, -1)
            ob['obstacle_vertices'][:self.obs_num] = cur_obs_vertices

        # 2. coordinates of bounding boxes (aabb)
        ob['obstacle_num'] = self.obs_num

        # 3. raw lidar point cloud
        # include everything (humans and obs) in lidar pc scan
        if self.config.ob_space.lidar_pc_include_humans:
            self.ray_test()
            ob['point_clouds'] = np.expand_dims(self.closest_hit_dist, axis=0)
        # only include obs
        # a. all obs in one lidar scan
        # b. seperate each obs for robot-obstacle attn
        else:
            self.ray_test_no_humans()
            ob['point_clouds'] = np.expand_dims(self.closest_hit_dist, axis=0)

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]

        # Stage 3: Acoustic edges observation — per-human response state for the policy.
        # Gated by self.human_visibility[i]: without this, a human occluded behind a wall
        # who happened to be acoustic_responding (e.g. triggered by an earlier beep while
        # still visible, or a compliant human just outside FOV) would have their exact
        # relative position handed to the network here, defeating the whole point of the
        # Stage-2 occlusion-inference module -- the robot could "ping" to reveal ground
        # truth instead of inferring it. Also reindexed through the same sort_order as
        # spatial_edges (computed above) so slot k in both arrays refers to the same human.
        if hasattr(self.config, 'acoustic') and self.config.acoustic.enabled:
            acoustic_obs = np.zeros((max(1, self.max_human_num), 4))
            for i in range(self.human_num):
                if (self.human_visibility[i] and
                        getattr(self.humans[i], 'acoustic_responding', False)):
                    rel_px = self.humans[i].px - self.robot.px
                    rel_py = self.humans[i].py - self.robot.py
                    if self.config.ob_space.human_state_frame == 'robot':
                        rel_pos = self.world_to_robot([rel_px, rel_py])
                        rel_px, rel_py = rel_pos[0], rel_pos[1]
                    acoustic_obs[self.humans[i].id] = [
                        1.0,  # is_responding flag
                        self.humans[i].acoustic_force_strength,
                        rel_px, rel_py
                    ]
            ob['acoustic_edges'] = acoustic_obs[sort_order]

            # Path B: what the robot heard. Reindexed through the same sort_order so
            # slot k lines up with spatial_edges/acoustic_edges slot k. A human who is
            # heard but NOT visible correctly lands in a slot whose spatial_edges row is
            # the [15,15,...] "unknown" sentinel -- which is exactly the intended signal:
            # "I cannot see this one, but I heard them roughly over there."
            # already computed above (before sorting) so it could feed spatial_edges
            ob['acoustic_returns'] = _ac_returns[sort_order]

        self.ob = ob

        return ob
