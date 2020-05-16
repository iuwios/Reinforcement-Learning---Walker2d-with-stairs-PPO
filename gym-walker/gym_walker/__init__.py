import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
        id='Walker-v0',
        entry_point='gym_walker.envs:Walker',
        max_episode_steps=1000,
)
