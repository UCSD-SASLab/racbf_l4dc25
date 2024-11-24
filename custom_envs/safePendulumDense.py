"""
Custom environment for safe pendulum with dense reward function
"""
try: 
  # When running inside module
  from utils import InvertedPendulumDeepreach, InvertedPendulumHJR
except: 
  # When running from outside module
  from custom_envs.utils import InvertedPendulumDeepreach, InvertedPendulumHJR
"""NOTE: JUST A TEST CHANGE LATER"""

# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""safePendulum domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

# reachability related imports 
from torch2jax import t2j, j2t
import jax 
import jax.numpy as jnp
import hj_reachability as hj

# import from local package
from custom_envs.safePendulum import SafeSwingUp, Physics, get_model_and_assets

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
import os 
CURR_FILE_PATH = os.path.dirname(__file__)


@SUITE.add('benchmarking')
def safeswingupdense(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns pendulum swingup task ."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = SafeSwingUpDense(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

class SafeSwingUpDense(SafeSwingUp):
  """
  SafeSwingUp task with denser reward functions
  """
  def get_reward(self, physics):
    base_reward = rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))
    if base_reward == 1.0:
      reward = base_reward
    else: 
      # densify reward 
      current_theta = (physics.named.data.qpos[0] + np.pi) % (2*np.pi) - np.pi
      reward = 1 - np.abs(current_theta)/np.pi
      reward = reward/2
      
    return reward

