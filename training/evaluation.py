import numpy as np
import torch

from crowd_sim.envs.utils.info import *
from training.networks import utils


def evaluate(actor_critic, eval_envs, num_processes, device, config, logging, test_args):
    """Evaluate the policy model (actor_critic) in multiple testing episodes.

        Parameters:
        actor_critic : torch.nn.Module
            The policy model to evaluate.
        eval_envs : VecEnv
            The vectorized environments for evaluation.
        num_processes : int
            Number of parallel environments to run.
        device : torch.device
            Device for running evaluation (CPU or CUDA).
        config : Config
            Configuration object with environment and training settings.
        logging : logging.Logger
            Logger for evaluation information.
        test_args : argparse.Namespace
            Additional testing arguments like visualization options.
        """

    test_size = config.env.test_size

    eval_episode_rewards = []

    # initialize the RNN hidden states
    eval_recurrent_hidden_states = {}
    if config.robot.policy in ['srnn', 'dsrnn_obs_pc', 'dsrnn_obs_vertex']:
        node_num = 1
        edge_num = actor_critic.base.human_num + 1 + actor_critic.base.obs_num
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num,
                                                                     config.SRNN.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           config.SRNN.human_node_rnn_size,
                                                                           device=device)

    else:
        eval_recurrent_hidden_states['rnn'] = torch.zeros(num_processes, 1, config.SRNN.human_node_rnn_size,
                                                          device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    # initialize testing metrics
    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    # Stage 3 / Path B acoustic metrics -- without these the acoustic mechanism cannot be
    # demonstrated at all, only its downstream effect on aggregate success rate.
    speak_counts = []            # beeps actually emitted per episode (cooldown-gated)
    occluded_heard_counts = []   # occluded humans localized via their vocal reply
    steps_per_episode = []

    success = 0
    collision = 0
    collision_human = 0
    collision_obs = 0
    collision_wall = 0

    timeout = 0
    too_close_ratios = []
    min_dist = []
    # centre-to-centre social metrics (see the step loop for why Danger.min_dist is not
    # usable -- it is a PyBullet surface distance that collapses to 0)
    step_separations = []     # every step's closest centre-to-centre gap
    intrusion_depths = []     # how far inside the comfort zone, per intruding step
    episode_min_seps = []     # worst separation reached in each episode
    _discomfort_dist = float(getattr(config.reward, 'discomfort_dist', 0.25))
    cumulative_rewards = []

    collision_cases = []
    collision_human_cases = []
    collision_obs_cases = []
    collision_wall_cases = []

    timeout_cases = []
    gamma = 0.99
    # .unwrapped is REQUIRED here, not cosmetic: envs[0].env is a gym OrderEnforcing
    # wrapper, and gym's Wrapper.__getattr__ raises AttributeError for any name starting
    # with '_' instead of forwarding it. Without .unwrapped, reads of the acoustic
    # detection counters (_last_episode_occluded_heard) silently returned the getattr
    # default of 0 -- which is why the echolocation metric reported 0 while detections
    # were in fact occurring. Public names like robot_is_speaking forwarded fine, which
    # is why beeps/episode looked correct and masked the problem.
    baseEnv = eval_envs.venv.envs[0].env.unwrapped

    # Optional inference-time safety filter. It is OFF unless cbf.enabled is set in the
    # run's config, and every configuration in this repository leaves it off; the filter
    # module itself is not part of this repository.
    _cbf = None
    if getattr(getattr(config, 'cbf', None), 'enabled', False):
        from cbf_filter import CBFFilter
        _cbf = CBFFilter(config,
                         horizon=int(getattr(config.cbf, 'horizon', 10)),
                         margin=float(getattr(config.cbf, 'margin', 0.10)),
                         obstacles=bool(getattr(config.cbf, 'obstacles', False)))
        logging.info('CBF shield ACTIVE: horizon=%d steps, margin=%.3f m, obstacles=%s',
                     _cbf.horizon, _cbf.margin, _cbf.obstacles)

    t = 0

    obs = eval_envs.reset()

    # the main testing loop
    for k in range(test_size):
        t += 1
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0

        # Capture THIS episode's obstacle geometry NOW, before any step. The vec-env
        # auto-resets when done=True, so reading obstacle_coord after the step loop
        # returns the NEXT episode's obstacles. Trajectory dumps made before 2026-09-17
        # had that bug: 9.4% of recorded robot positions sat inside "obstacles" that were
        # never there.
        try:
            import numpy as _np
            _episode_obs = _np.asarray(baseEnv.obstacle_coord, dtype=float).tolist()
        except Exception as _e:
            _episode_obs = None
            print('WARNING: obstacle capture failed at episode start (%r)' % (_e,))

        global_time = 0.0
        path = 0.0
        too_close = 0.
        ep_speaks = 0
        episode_min_sep = float('inf')

        last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py

        # Per-step trajectory record for offline analysis of individual episodes; only
        # written to disk when env.traj_dump is set (see the dump after the outcome branch).
        traj_record = []

        while not done:
            stepCounter = stepCounter + 1
            # given observation, forward the robot policy to get action
            if not test_args.dwa:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else: # for DWA
                u, predicted_trajectory, curr_state, action = actor_critic.predict(eval_envs.venv.envs[0].env)

            if _cbf is not None:
                _a = action.item() if isinstance(action, torch.Tensor) else int(action)
                _safe = _cbf.filter(baseEnv, _a)
                if _safe != _a:
                    action = (torch.tensor([[_safe]], device=action.device)
                              if isinstance(action, torch.Tensor) else _safe)

            if hasattr(test_args, 'random_speak') and test_args.random_speak:
                import random
                if isinstance(action, torch.Tensor):
                    a_val = action.item()
                    base_a = a_val % 9
                    new_a = base_a + 9 if random.random() < 0.5 else base_a
                    action = torch.tensor([[new_a]], device=action.device)
                else:
                    a_val = int(action)
                    base_a = a_val % 9
                    new_a = base_a + 9 if random.random() < 0.5 else base_a
                    action = new_a

            if not done:
                global_time = baseEnv.global_time
                # react_to_robot is per-human, drawn at spawn from robot.visible_prob
                # (crowd_sim_var_human.py:285); recording it lets an offline analysis tell
                # pedestrians that could perceive the robot from those that could not.
                traj_record.append((
                    global_time,
                    baseEnv.robot.px, baseEnv.robot.py,
                    baseEnv.robot.vx, baseEnv.robot.vy,
                    baseEnv.robot.theta,
                    [(h.px, h.py, h.vx, h.vy, h.radius,
                      bool(getattr(h, 'react_to_robot', False))) for h in baseEnv.humans],
                ))
            if test_args.visualize:
                eval_envs.render()

            # step the environment to get reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            # Tally acoustic activity from the env itself (baseEnv is the unwrapped
            # CrowdSim3DTbObs, so these are the same attributes step() just set).
            if getattr(baseEnv, 'robot_is_speaking', False):
                ep_speaks += 1

            # NOTE (inherited from upstream HEIGHT): on the terminal step the vectorised env
            # has already reset, so obs here is the NEXT episode's first observation and this
            # adds the jump to the next start position to the logged path length.
            path = path + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)

            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            rewards.append(rew)

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            # Centre-to-centre separation and intrusion DEPTH.
            #
            # The Danger.min_dist above comes from PyBullet getClosestPoints()[8], which
            # is a SURFACE-to-surface distance -- it collapses to ~0 whenever bodies are
            # nearly touching, which is why it logs as 0.00 and carries no information.
            # These two measure something reportable instead:
            #   separation = centre-to-centre distance minus both radii (the standard
            #                "minimum separation" number in social-navigation tables)
            #   depth      = how far INSIDE the comfort zone the robot got, so severity
            #                is distinguished from mere frequency
            # NOTE: on the terminal step baseEnv has already reset, so this block then reads
            # the next episode's start state; separations and depths include that sample.
            try:
                _rx, _ry = baseEnv.robot.px, baseEnv.robot.py
                _rr = baseEnv.robot.radius
                _sep_step = float('inf')
                for _h in baseEnv.humans:
                    _d = np.hypot(_rx - _h.px, _ry - _h.py) - (_rr + _h.radius)
                    if _d < _sep_step:
                        _sep_step = _d
                if _sep_step < float('inf'):
                    step_separations.append(_sep_step)
                    if _sep_step < _discomfort_dist:
                        intrusion_depths.append(_discomfort_dist - _sep_step)
                    if _sep_step < episode_min_sep:
                        episode_min_sep = _sep_step
            except Exception:
                pass

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        path_lengths.append(path)
        too_close_ratios.append(too_close / stepCounter * 100)
        if episode_min_sep < float('inf'):
            episode_min_seps.append(episode_min_sep)
        speak_counts.append(ep_speaks)
        # unique occluded humans localized by audio this episode (env-side set, so a
        # human vocalizing across several steps counts once)
        occluded_heard_counts.append(getattr(baseEnv, '_last_episode_occluded_heard', 0))
        steps_per_episode.append(stepCounter)

        # Dump trajectories for offline analysis: ALL collisions, plus a 1-in-N sample of
        # SUCCESSES as a control set, which keeps the footprint at a few hundred files.
        # OFF by default; enable per run with `env.traj_dump = True` in that run's config.
        _outcome = infos[0]['info']
        _succ_stride = int(getattr(config.env, 'traj_success_stride', 8))
        if not getattr(config.env, 'traj_dump', False):
            _succ_stride = -1  # disables the success branch; collision branch guarded below
        _dump = getattr(config.env, 'traj_dump', False) and (
            isinstance(_outcome, (CollisionHuman, CollisionObs, CollisionWall)) or (
            isinstance(_outcome, ReachGoal) and _succ_stride > 0
            and (success % _succ_stride == 0)))
        if _dump:
            import os as _os, pickle as _pickle
            _tdir = _os.path.join(config.training.output_dir, 'trajectories')
            _os.makedirs(_tdir, exist_ok=True)
            with open(_os.path.join(_tdir, 'ep_%05d.pkl' % k), 'wb') as _f:
                # Obstacle geometry is dumped ONCE per episode (it is static), so a replay
                # can check recorded or counterfactual robot positions against it.
                # obstacle_coord is the env attribute holding world-frame rectangle
                # vertices; cur_obs_vertices in generate_ob() is a robot-frame LOCAL, not
                # an attribute, so reading it here would silently yield None.
                # Use the geometry captured at episode START (see top of loop). Do NOT
                # re-read baseEnv.obstacle_coord here: the env has already reset.
                _obs = _episode_obs
                _pickle.dump({
                    'episode': k,
                    'outcome': type(_outcome).__name__,
                    'robot_radius': baseEnv.robot.radius,
                    # config.robot.v_max (0.5), NOT robot.v_pref (which is 1.0 and is
                    # the ORCA preferred-speed field, not the actuator limit).
                    'robot_v_max': float(getattr(baseEnv.config.robot, 'v_max', 0.5)),
                    'robot_v_min': float(getattr(baseEnv.config.robot, 'v_min', -0.5)),
                    'robot_w_max': float(getattr(baseEnv.config.robot, 'w_max', 1.0)),
                    'time_step': baseEnv.time_step,
                    'obstacle_vertices': _obs,
                    'arena_size': getattr(baseEnv.config.sim, 'arena_size', None),
                    'steps': traj_record,
                }, _f)

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], CollisionHuman):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_human += 1
            collision_human_cases.append(k)
            print('Collision with human')
        elif isinstance(infos[0]['info'], CollisionObs):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_obs += 1
            collision_obs_cases.append(k)
            print('Collision with obstacle')
        elif isinstance(infos[0]['info'], CollisionWall):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_wall += 1
            collision_wall_cases.append(k)
            print('Collision with wall')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))

    # after all testing episodes are done,
    # calculate and log results
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size

    collision_human_rate = collision_human / test_size
    collision_obs_rate = collision_obs / test_size
    collision_wall_rate = collision_wall / test_size
    print(success, collision, timeout)
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else baseEnv.time_limit  # baseEnv.env.time_limit

    extra_info = ''
    phase = 'test'
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}'.
        format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time))
    logging.info(
        'collision rate with humans: {:.2f}, with obstacles: {:.2f}, with walls: {:.2f}, '.
        format(collision_human_rate, collision_obs_rate, collision_wall_rate))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('average intrusion ratio: %.2f and average minimal distance during intrusions: %.2f',
                     np.mean(too_close_ratios), avg_min_dist)

        # Per-episode spread, so intrusion ratio can carry a significance test the way
        # success rate does. The bare mean above is not comparable across systems without
        # it: a 12% relative difference means nothing until you know the per-episode SD.
        _tcr = np.asarray(too_close_ratios, dtype=float)
        if _tcr.size:
            _sem = _tcr.std(ddof=1) / np.sqrt(_tcr.size) if _tcr.size > 1 else float('nan')
            logging.info('intrusion ratio (per-episode): %.4f +/- %.4f  sem=%.4f  n=%d  '
                         'median=%.4f  frac_zero=%.3f',
                         _tcr.mean(), _tcr.std(ddof=1) if _tcr.size > 1 else float('nan'),
                         _sem, _tcr.size, np.median(_tcr), float((_tcr == 0).mean()))
        # Social metrics from centre positions: each separation is the centre-to-centre
        # distance minus both radii (the label below is kept for scripts that parse it).
        # The surface-distance min_dist below is ~0 by construction and says nothing.
        _sep = np.asarray(step_separations, dtype=float)
        if _sep.size:
            logging.info('separation centre-to-centre (per-step): mean=%.4f  p05=%.4f  '
                         'p01=%.4f  min=%.4f  n=%d',
                         _sep.mean(), np.percentile(_sep, 5), np.percentile(_sep, 1),
                         _sep.min(), _sep.size)
        _ems = np.asarray(episode_min_seps, dtype=float)
        if _ems.size:
            logging.info('episode-worst separation: %.4f +/- %.4f  median=%.4f  n=%d',
                         _ems.mean(), _ems.std(ddof=1) if _ems.size > 1 else float('nan'),
                         np.median(_ems), _ems.size)
        _dep = np.asarray(intrusion_depths, dtype=float)
        if _dep.size:
            logging.info('intrusion DEPTH (m inside the %.2f m comfort zone): '
                         'mean=%.4f +/- %.4f  max=%.4f  n_steps=%d',
                         _discomfort_dist, _dep.mean(),
                         _dep.std(ddof=1) if _dep.size > 1 else float('nan'),
                         _dep.max(), _dep.size)
        else:
            logging.info('intrusion DEPTH: no steps inside the comfort zone')

        _md = np.asarray(min_dist, dtype=float)
        if _md.size:
            logging.info('min dist during intrusions (per-event): %.4f +/- %.4f  n=%d  min=%.4f',
                         _md.mean(), _md.std(ddof=1) if _md.size > 1 else float('nan'),
                         _md.size, _md.min())
        else:
            logging.info('min dist during intrusions: NO EVENTS RECORDED (min_dist list empty)')

        # Raw per-episode values, so a paired/unpaired test can be run offline without
        # re-running the evaluation.
        try:
            import json as _json, os as _os
            _dump = _os.path.join('logs', 'intrusion_%s_%s.json'
                                  % (phase, _os.path.basename(str(config.training.output_dir)).strip("'")))
            _os.makedirs('logs', exist_ok=True)
            with open(_dump, 'w') as _f:
                _json.dump({'too_close_ratios': _tcr.tolist(),
                            'min_dist': _md.tolist(),
                            'episode_min_seps': _ems.tolist() if _ems.size else [],
                            'intrusion_depths': _dep.tolist() if _dep.size else [],
                            'separation_p05': float(np.percentile(_sep, 5)) if _sep.size else None,
                            'path_lengths': [float(x) for x in path_lengths]}, _f)
            logging.info('per-episode intrusion data written to %s', _dump)
        except Exception as _e:
            logging.info('could not write per-episode intrusion dump: %s', _e)

    if _cbf is not None:
        _n, _o, _r = _cbf.stats()
        logging.info('CBF: %d steps checked, %d overridden (%.2f%%)', _n, _o, _r)
        if _o == 0:
            logging.info('CBF WARNING: filter NEVER fired -- a null result here says '
                         'nothing about shielding, only that the margin/horizon are inert.')

    logging.info(
        '{:<5} {}has average path length: {:.2f}'.
        format(phase.upper(), extra_info, sum(path_lengths) / test_size))

    # Acoustic-extension metrics (only non-trivial when acoustic.enabled is set; every
    # configuration in this repository leaves it off). Reported with standard deviations
    # so that a counter that never moves shows up as zero variance.
    if speak_counts:
        speak_arr = np.array(speak_counts, dtype=float)
        heard_arr = np.array(occluded_heard_counts, dtype=float)
        steps_arr = np.array(steps_per_episode, dtype=float)
        total_speaks = speak_arr.sum()
        logging.info(
            'ACOUSTIC beeps/episode: %.2f +/- %.2f | beeps per 100 steps: %.2f',
            speak_arr.mean(), speak_arr.std(), 100.0 * total_speaks / max(1.0, steps_arr.sum()))
        logging.info(
            'ACOUSTIC occluded-humans-localized-by-audio per episode: %.2f +/- %.2f | total: %d',
            heard_arr.mean(), heard_arr.std(), int(heard_arr.sum()))
        logging.info(
            'ACOUSTIC information yield (occluded humans localized per beep): %.3f',
            heard_arr.sum() / total_speaks if total_speaks > 0 else float('nan'))

    # Report dispersion on the headline metrics too, so arms differing by ~0.4s of nav
    # time can be compared against the actual spread instead of eyeballed.
    if success_times:
        st = np.array(success_times, dtype=float)
        logging.info('nav time (successes): %.2f +/- %.2f over n=%d', st.mean(), st.std(), st.size)
    pl = np.array(path_lengths, dtype=float)
    logging.info('path length: %.2f +/- %.2f', pl.mean(), pl.std())
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Collision with Human cases: ' + ' '.join([str(x) for x in collision_human_cases]))
    logging.info('Collision with Obstacle cases: ' + ' '.join([str(x) for x in collision_obs_cases]))
    logging.info('Collision with Wall cases: ' + ' '.join([str(x) for x in collision_wall_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
