"""
Custom environment for safe pendulum
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

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
import os 
CURR_FILE_PATH = os.path.dirname(__file__)


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('safePendulum.xml'), common.ASSETS
  from dm_control.utils import io as resources
  custom_envs_folder = CURR_FILE_PATH #os.path.dirname(os.path.dirname(__file__))
  model_filename = os.path.join(custom_envs_folder, "safePendulum.xml")
  print("Loading environment model from: ", model_filename)
  return resources.GetResource(model_filename), common.ASSETS

# from dm_control.suite.pendulum import get_model_and_assets # NOTE: Load standard pendulum xml file 

@SUITE.add('benchmarking')
def safeswingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns pendulum swingup task ."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = SafeSwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Pendulum domain."""

  def pole_vertical(self):
    """Returns vertical (z) component of pole frame."""
    return self.named.data.xmat['pole', 'zz']

  def angular_velocity(self):
    """Returns the angular velocity of the pole."""
    return self.named.data.qvel['hinge'].copy()

  def pole_orientation(self):
    """Returns both horizontal and vertical components of pole frame."""
    return self.named.data.xmat['pole', ['zz', 'xz']]


class SafeSwingUp(base.Task):
  """A Pendulum `Task` to swing up and balance the pole."""

  def __init__(self, random=None):
    """Initialize an instance of `Pendulum`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    # self.set_unsafe_region(unsafe_theta_min=np.pi/2 - np.pi/4, unsafe_theta_max=np.pi/2 - np.pi/8)
    self.set_unsafe_region(unsafe_theta_min=np.pi/8, unsafe_theta_max=np.pi/4)
    self.length = 0.5 #1.0 # NOTE: may want to change based on how you modify things
    self.mass = 1.0
    self.damping = 0.1
    self.setup_hj_reachability()

    super().__init__(random=random)

  def set_unsafe_region(self, unsafe_theta_min, unsafe_theta_max): 
    """
    Set the unsafe region: 
    args: 
      - unsafe_theta_min: theta (in radians) describing start of unsafe region 
      - unsafe_theta_max: theta (in radians) describing end of unsafe region
    """
    self.unsafe_theta_min = unsafe_theta_min
    self.unsafe_theta_max = unsafe_theta_max

  def setup_hj_reachability(self):
    gravity = 9.81
    length = self.length 
    mass = self.mass 

    unsafe_theta_min = self.unsafe_theta_min
    unsafe_theta_max = self.unsafe_theta_max
    min_torque = -1.0
    max_torque = 1.0 

    max_theta_dist = 0.01
    max_thetadot_dist = 0 
    tMin = 0.0
    tMax = 10

    theta_range = [-np.pi, np.pi]
    thetadot_range = [-10, 10]
    grid_resolution = (201, 101)
    time_resolution = 101

    self.inv_pendulum_deepreach = InvertedPendulumDeepreach(gravity=gravity, 
                                                            length=length, 
                                                            mass=mass, 
                                                            unsafe_theta_min=unsafe_theta_min,
                                                            unsafe_theta_max=unsafe_theta_max, 
                                                            min_torque=min_torque, 
                                                            max_torque=max_torque,
                                                            max_theta_dist=max_theta_dist, 
                                                            max_thetadot_dist=max_thetadot_dist,  
                                                            damping=self.damping,
                                                            tMin=tMin, 
                                                            tMax=tMax)
    self.inv_pendulum_hjr = InvertedPendulumHJR(torch_dynamics=self.inv_pendulum_deepreach, 
                                                    gravity=gravity, 
                                                    length=length,
                                                    mass=mass, 
                                                    unsafe_theta_min=unsafe_theta_min, 
                                                    unsafe_theta_max=unsafe_theta_max, 
                                                    min_torque=min_torque, 
                                                    max_torque=max_torque,
                                                    max_theta_dist=max_theta_dist, 
                                                    max_thetadot_dist=max_thetadot_dist, 
                                                    damping=self.damping,
                                                    tMin=tMin, 
                                                    tMax=tMax)
    # HJ Reachability Solver Settings
    state_domain = hj.sets.Box(np.array([theta_range[0], thetadot_range[0]]), 
                                np.array([theta_range[1], thetadot_range[1]]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, 
                                                                        grid_resolution, 
                                                                        periodic_dims=0)
    sdf_values = t2j(self.inv_pendulum_hjr.torch_dynamics.boundary_fn(j2t(grid.states)))
    times = jnp.linspace(tMin, -tMax, time_resolution)
    initial_values = sdf_values 
    solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    # Solve HJI value function
    all_values = hj.solve(solver_settings, self.inv_pendulum_hjr, grid, times, initial_values, progress_bar=True)
    target_values = all_values[-1]
    diffs = -jnp.diff(all_values, axis=0).mean(axis=(1,2))
    desired_diff_epsilon = 1e-5
    assert(diffs[-1] < desired_diff_epsilon)

    # Create general safe environment attributes to be used externally 
    self.hjr_object = self.inv_pendulum_hjr
    self.deepreach_object = self.inv_pendulum_deepreach
    self.hjr_grid = grid 
    self.hjr_all_values = all_values 
    self.hjr_times = times 

    self.hjr_target_values = target_values 
    hjr_grid_interpolator_v = jax.vmap(self.hjr_grid.interpolate, in_axes=(None, 0))
    self.hjr_state_to_value = lambda state: hjr_grid_interpolator_v(jnp.array(self.hjr_all_values[-1]), state[None])[0]

    return 
  
  def obs_to_cbfstate(self, obs): 
    """
    To be used by the SAC agent for a CBF safety filter
    args: 
      - obs: torch tensor potentially on GPU
    returns: 
      - state: numpy array 
    """
    # convert observation to state 
    # NOTE: in HJR 0 degrees means pendulum is at the top, np.pi is at the bottom angle increases on the left (counter clockwise)
    # NOTE: As a result you have to take arctan of (x/y) instead of typical (y/x) here
    obs_cpu = obs.cpu().numpy()
    theta = np.arctan2(obs_cpu[..., 1], obs_cpu[..., 0])
    theta_dot = obs_cpu[..., 2]
    state = np.array([theta[0], theta_dot[0]])
    return state 

  def is_unsafe(self, obs):
    """
    Returns boolean if the pendulum is in the unsafe region
    args: 
      - obs: observation 
        - 'orientation'
        - 'velocity'
    """
    # NOTE: for our purposes: 0 is on top, pi/2 is on the right, pi is on the bottom, -pi/2 is on the left
    vertical, horizontal = obs['orientation']
    theta = np.arctan2(horizontal, vertical)
    if self.unsafe_theta_min <= theta and theta <= self.unsafe_theta_max: 
      return True 
    else: 
      return False 
    
  def is_unsafe_physics(self, physics): 
    """
    Returns boolean if the pendulum is in the unsafe region
    args: 
      - obs: observation 
        - 'orientation'
        - 'velocity'
    """
    # NOTE: for our purposes: 0 is on top, pi/2 is on the right, pi is on the bottom, -pi/2 is on the left
    current_theta = (physics.named.data.qpos[0] + np.pi) % (2*np.pi) - np.pi
    if self.unsafe_theta_min <= current_theta and current_theta <= self.unsafe_theta_max: 
      return True 
    else: 
      return False 

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Pole is set to a random angle between [-pi, pi).

    Args:
      physics: An instance of `Physics`.

    """
    # Safety modification: do not initialize in an unsafe region
    max_counter = 1000
    counter = 0 
    found_start = False 

    while not found_start: 
      start_state = np.array([self.random.uniform(-np.pi, np.pi), 0])
      start_state_value = self.hjr_state_to_value(start_state)
      if start_state_value >= 0: 
        # safe 
        found_start = True 
      else: 
        counter += 1
        if counter > max_counter: 
          # Force 0 and print that it occurred 
          start_state = np.array([np.pi, 0.0])
          print("\n\n\n\nMax counter exceeded: forcing to ", start_state)
          print("\n\n\n")
          found_start = True 

    physics.named.data.qpos['hinge'] = start_state[0] # self.random.uniform(-np.pi, np.pi)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation.

    Observations are states concatenating pole orientation and angular velocity
    and pixels from fixed camera.

    Args:
      physics: An instance of `physics`, Pendulum physics.

    Returns:
      A `dict` of observation.
    """
    obs = collections.OrderedDict()
    obs['orientation'] = physics.pole_orientation()
    obs['velocity'] = physics.angular_velocity()

    if self.is_unsafe(obs=obs): 
      physics.named.model.geom_rgba['pole'] = [1, 0, 0, 1] # force red for now
      physics.named.model.geom_rgba['mass'] = [1, 0, 0, 1] # force red for now
    else: 
      physics.named.model.geom_rgba['pole'] = [0.5, 0.5, 0.5, 1] # default back to beige
      physics.named.model.geom_rgba['mass'] = [0.5, 0.5, 0.5, 1] # default back to beige

    return obs

  def get_reward(self, physics):
    return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))
