import numpy as np

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        # whether the human is a static obstacle (part of wall) or a moving agent
        self.isObstacle = False
        self.isObstacle_period = np.inf
        # whether the human reacts to the robot
        self.react_to_robot = False
        # route of this human, only exists in constrained env
        self.route = None

        # Stage 3: Acoustic response state
        self.acoustic_responding = False       # Is this human currently responding to speech?
        self.acoustic_response_timer = 0       # Countdown timer for response duration
        self.acoustic_force_dir = np.array([0., 0.])  # Direction to move away from robot
        self.acoustic_force_strength = 0.0     # How strongly this human should respond
        self.acoustic_signal_strength = 0.0    # Raw Gaussian signal (0-1) at trigger time
        self.acoustic_compliant = True         # Does this human comply? (randomized per episode)
        self.acoustic_pending = False          # Triggered by a beep, waiting out response_delay
        self.acoustic_delay_timer = 0           # Countdown before the response actually starts
        self.acoustic_returning = False        # Path B: currently vocalizing a reply ("sorry!")
        self.acoustic_return_timer = 0         # Countdown for how long that reply stays audible
        self.acoustic_belief_world = None      # Remembered WORLD-frame audio fix (x, y)
        self.acoustic_belief_timer = 0         # How long that memory remains valid
        self.acoustic_belief_quality = 0.0     # (vestigial; superseded by belief_err)
        self.acoustic_belief_err = 0.0         # Expected position error (m) of that fix
        self.acoustic_belief_n = 0             # Fixes averaged during the current utterance
        self.original_gx = None                # Original goal X to return to after response
        self.original_gy = None                # Original goal Y to return to after response

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_joint_state(self, ob):
        """Act from a pre-constructed JointState (for lookahead simulation)."""
        action = self.policy.predict(ob)
        return action
