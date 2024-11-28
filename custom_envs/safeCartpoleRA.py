"""
Safe Cartpole with Reach Avoid
"""

"""
What you need to do:
1. @SUITE.add('benchmarking') swingup function 
2. Import the Balance task and hten modify it:
    - Particularly change the setup_hjr function so that you have the reach avoid value function 
"""

"""
Safe Cartpole environment
"""
try: 
  # When running inside module
  from utils import BaseCartPoleDeepreach, BaseCartPoleHJR
  from safeCartpole import Balance as BalanceSafeCartpole
  from safeCartpole import get_model_and_assets, _make_model 
except: 
  # When running from outside module
  from custom_envs.utils import BaseCartPoleDeepreach, BaseCartPoleHJR
  from custom_envs.safeCartpole import Balance as BalanceSafeCartpole
  from custom_envs.safeCartpole import get_model_and_assets, _make_model 


"""Safe Cartpole RA domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np
import os 

import hj_reachability as hj
from torch2jax import t2j, j2t
import jax 
import jax.numpy as jnp 

import torch

_DEFAULT_TIME_LIMIT = 15 #20 #10
SUITE = containers.TaggedTasks()
CURR_FILE_PATH = os.path.dirname(__file__)


################################################################################################################################################################
import torch 
def cartpole_sdf(state, 
                 unsafe_x_min, unsafe_x_max, unsafe_vel_max, unsafe_theta_min, unsafe_theta_max, unsafe_thetadot_max, 
                 unsafe_theta_in_range): 

    # unsafe_theta_in_range: if True then the theta range describes what is unsafe
    if isinstance(state, torch.Tensor):
        state = state 
    else: 
        state = torch.tensor(state)

    use_unsafe_theta = True 
    if unsafe_theta_min == unsafe_theta_max: 
        use_unsafe_theta = False 

    x = state[..., 0]
    theta = state[..., 1]
    xdot = state[..., 2]
    thetadot = state[..., 3]

    # Unsafe x: in range is safe 
    unsafe_x = torch.zeros(x.shape).to(state.device)
    greater_than_x = torch.where(x > unsafe_x_max)
    less_than_x = torch.where(x < unsafe_x_min)
    in_range_x = torch.where((unsafe_x_min < x) & (x < unsafe_x_max))
    unsafe_x[greater_than_x] = (unsafe_x_max - x)[greater_than_x] # negative unsafe 
    unsafe_x[less_than_x] = (x - unsafe_x_min)[less_than_x] # negative unsafe
    unsafe_x[in_range_x] = torch.min(x - unsafe_x_min, unsafe_x_max - x)[in_range_x]

    # Unsafe velocity: in range is safe 
    unsafe_xdot = torch.zeros(xdot.shape).to(state.device)
    greater_than_xdot = torch.where(xdot > unsafe_vel_max)
    less_than_xdot = torch.where(xdot < -unsafe_vel_max)
    in_range_xdot = torch.where((-unsafe_vel_max < xdot) & (xdot < unsafe_vel_max))
    unsafe_xdot[greater_than_xdot] = (unsafe_vel_max - xdot)[greater_than_xdot] # negative unsafe
    unsafe_xdot[less_than_xdot] = (xdot - (-1 * unsafe_vel_max))[less_than_xdot] # negative unsafe
    unsafe_xdot[in_range_xdot] = torch.min(xdot - (-1 * unsafe_vel_max), unsafe_vel_max - xdot)[in_range_xdot]

    if use_unsafe_theta: 

        unsafe_theta = torch.zeros(theta.shape).to(state.device)
        greater_than_theta = torch.where(theta > unsafe_theta_max) 
        less_than_theta = torch.where(theta < unsafe_theta_min)
        in_range_theta = torch.where((unsafe_theta_min < theta) & (theta < unsafe_theta_max))
        
        if unsafe_theta_in_range: 
            # Unsafe Theta: in range is unsafe
            unsafe_theta[greater_than_theta] = (theta - unsafe_theta_max)[greater_than_theta]
            unsafe_theta[less_than_theta] = (unsafe_theta_min - theta)[less_than_theta]
            unsafe_theta[in_range_theta] = torch.min(unsafe_theta_min - theta, theta - unsafe_theta_max)[in_range_theta] # negative unsafe
        else: 
            # Safe Theta: in range, Unsafe theta: out of range 
            unsafe_theta[greater_than_theta] = (unsafe_theta_max - theta)[greater_than_theta]
            unsafe_theta[less_than_theta] = (theta - unsafe_theta_min)[less_than_theta]
            unsafe_theta[in_range_theta] = torch.min(theta - unsafe_theta_min, unsafe_theta_max - theta)[in_range_theta]

        # TODO: NOTE: might need to change 
        unsafe_vals = torch.min(unsafe_x, torch.min(unsafe_xdot, unsafe_theta))
    else: 
        unsafe_vals = torch.min(unsafe_x, unsafe_xdot)
    
    # Unsafe thetadot: in range is safe 
    unsafe_thetadot = torch.zeros(thetadot.shape).to(state.device)
    greater_than_thetadot = torch.where(thetadot > unsafe_thetadot_max)
    less_than_thetadot = torch.where(thetadot < -unsafe_thetadot_max)
    in_range_thetadot = torch.where((-unsafe_thetadot_max < thetadot) & (thetadot < unsafe_thetadot_max))
    unsafe_thetadot[greater_than_thetadot] = (unsafe_thetadot_max - thetadot)[greater_than_thetadot] # negative unsafe
    unsafe_thetadot[less_than_thetadot] = (thetadot - (-1 * unsafe_thetadot_max))[less_than_thetadot] # negative unsafe
    unsafe_thetadot[in_range_thetadot] = torch.min(thetadot - (-1 * unsafe_thetadot_max), unsafe_thetadot_max - xdot)[in_range_thetadot]
    
    unsafe_vals = torch.min(unsafe_vals, unsafe_thetadot)
    # import pdb; pdb.set_trace()
    return unsafe_vals

################################################################################################################################################################


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = BalanceRA(swing_up=True, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the sparse reward variant of the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = BalanceRA(swing_up=True, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cartpole domain."""

  def cart_position(self):
    """Returns the position of the cart."""
    return self.named.data.qpos['slider'][0]

  def angular_vel(self):
    """Returns the angular velocity of the pole."""
    return self.data.qvel[1:]

  def pole_angle_cosine(self):
    """Returns the cosine of the pole angle."""
    return self.named.data.xmat[2:, 'zz']

  def bounded_position(self):
    """Returns the state, with pole angle split into sin/cos."""
    return np.hstack((self.cart_position(),
                      self.named.data.xmat[2:, ['zz', 'xz']].ravel()))


class BalanceRA(base.Task):
  """A Cartpole `Task` to balance the pole.

  State is initialized either close to the target configuration or at a random
  configuration.
  """
  def __init__(self, swing_up, sparse, random=None):
    """Initializes an instance of `Balance`.

    Args:
      swing_up: A `bool`, which if `True` sets the cart to the middle of the
        slider and the pole pointing towards the ground. Otherwise, sets the
        cart to a random position on the slider and the pole to a random
        near-vertical position.
      sparse: A `bool`, whether to return a sparse or a smooth reward.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """

    self.init_in_target = True # Force environment to initialize in target/reach set

    # super().__init__(swing_up=swing_up, sparse=sparse, random=random)
    self._sparse = sparse
    self._swing_up = swing_up
    self.setup_hj_reachability()

    # Times for CBF 
    self.max_time = _DEFAULT_TIME_LIMIT 
    self.dt = 0.01
    self.max_iterations = int(self.max_time / self.dt)
    self.return_time = -self.hjr_times[-1] # time to reach target set 

    super().__init__(random=random)
    return 
  
  def set_init_in_target(self, init_in_target): 
    """
    Function to change init_in_target attribute
    Default: True : forces the environment to initialize in the target reach (reach set)
             False: environment can initialize in any safe region (positive value for the target time reach avoid value function)
    """
    self.init_in_target = init_in_target 
    return
  
  def set_environment_state(self, physics, state):
    # Force set the environment state 
    if isinstance(state, torch.Tensor):
      state = state.cpu().detach().numpy()
    else: 
      state = np.array(state)
    
    x, theta, xdot, thetadot = state 
    physics.named.data.qpos[0] = x # cart position
    physics.named.data.qpos[1] = theta # pole position
    physics.named.data.qvel[0] = xdot # cart velocity
    physics.named.data.qvel[1] = thetadot # pole velocity

    # Update the physics simulation to reflect changes
    # physics.forward() # NOTE: TODO: Might want to remove this ? or add this ? 

    updated_obs = self.get_observation(physics)

    updated_obs = [updated_obs['position'][0], updated_obs['position'][1], updated_obs['position'][2], 
                   updated_obs['velocity'][0], updated_obs['velocity'][1]]
    return updated_obs 
  
  def get_environment_state(self, physics):
    # Get the environment state - specifically for resetting 
    x = physics.named.data.qpos[0]
    theta = (physics.named.data.qpos[1] + np.pi)%(2*np.pi) - np.pi
    xdot = physics.named.data.qvel[0]
    thetadot = physics.named.data.qvel[1]
    return np.array([x, theta, xdot, thetadot])

  def is_unsafe(self, physics): 
    """
    Returns boolean if the cartpole is in the unsafe region
    """
    state = self.get_environment_state(physics=physics)

    return self.cartpole_deepreach.is_unsafe(state=np.array([state], dtype=np.float32))  

  def avoid_sdf(self, state):
    # Avoid sdf function 
    self.avoid_unsafe_x_min     = -1.5 
    self.avoid_unsafe_x_max     = 1.5 
    self.avoid_unsafe_vel_max   = 20 
    
    self.avoid_unsafe_theta_min = np.pi/8
    self.avoid_unsafe_theta_max =  np.pi/4
    self.avoid_unsafe_thetadot_max = 20

    self.avoid_unsafe_theta_in_range = True # True = specified theta range is unsafe

    return cartpole_sdf(state, self.avoid_unsafe_x_min, self.avoid_unsafe_x_max, self.avoid_unsafe_vel_max, self.avoid_unsafe_theta_min, self.avoid_unsafe_theta_max, self.avoid_unsafe_thetadot_max, self.avoid_unsafe_theta_in_range)
  
  def reach_sdf(self, state):
    # Reach sdf function 
    self.reach_unsafe_x_min     = -1.1 #0.15
    self.reach_unsafe_x_max     = -0.8 #0.15
    self.reach_unsafe_vel_max   = 0.1
    
    # reach_unsafe_theta_min = np.pi - 0.25
    # reach_unsafe_theta_max =  np.pi + 0.25
    # reach_unsafe_thetadot_max = 0.25
    # reach_unsafe_theta_in_range = False # False = everywhere outside theta range is unsafe 

    self.reach_unsafe_theta_min = -np.pi + 0.25
    self.reach_unsafe_theta_max =  np.pi - 0.25
    self.reach_unsafe_thetadot_max = 0.25
    self.reach_unsafe_theta_in_range = True # True = specified theta range is unsafe
    
    return cartpole_sdf(state, self.reach_unsafe_x_min, self.reach_unsafe_x_max, self.reach_unsafe_vel_max, self.reach_unsafe_theta_min, self.reach_unsafe_theta_max, self.reach_unsafe_thetadot_max, self.reach_unsafe_theta_in_range)

  def sample_state_in_target_set(self): 
      """
      Function to sample state in the reach set. 
      To be used in initializing the environent when force_target_init is True. 
      args: 
        - physics: physics object
      returns: 
        - state: [x, theta, xdot, thetadot] in the reach set 
      """
      self.reach_sdf(state=np.array([[0, 0, 0, 0]], dtype=np.float32)) # force initialization of self.reach_* attributes

      # NOTE: will need to change this if you change the reach set - general way is to sample uniformly and then use the sdf to filter - but inefficient
      x = np.random.uniform(self.reach_unsafe_x_min, self.reach_unsafe_x_max)
      if not self.reach_unsafe_theta_in_range: 
        theta = np.random.uniform(self.reach_unsafe_theta_min, self.reach_unsafe_theta_max)
      else: 
        theta_neg = np.random.uniform(-np.pi, self.reach_unsafe_theta_min)
        theta_pos = np.random.uniform(self.reach_unsafe_theta_max, np.pi)
        theta = np.random.choice([theta_neg, theta_pos])
      xdot = np.random.uniform(-self.reach_unsafe_vel_max, self.reach_unsafe_vel_max)
      thetadot = np.random.uniform(-self.reach_unsafe_thetadot_max, self.reach_unsafe_thetadot_max)

      state = np.array([[x, theta, xdot, thetadot]], dtype=np.float32)
      assert(self.reach_sdf(state) >= 0)

      return state[0]

  def setup_hj_reachability(self): 

    # Unsafe Sliver Configuration
    # For debugging purposes
    re_compute_hjr = False #True #True # False
    hjr_filename = "rl_safeCartpoleRA_hjr_values.npy"
    hjr_filename = os.path.join(CURR_FILE_PATH, hjr_filename)

    # Dynamics attributes
    gravity= 9.8 #-9.8
    umax=10
    length=1.0 #0.5
    mass_cart=1.0
    mass_pole=0.1

    # Disturbance bounds
    x_dist        = 0.02 #0.0
    theta_dist    = 0.02 #0.0
    vel_dist      = 0.05 #0.02 #0.2
    thetadot_dist = 0.05 #0.02 #0.2

    # Timesteps 
    tMin          = 0.0
    tMax          = 15.0 

    # HJR State Space Range
    x_range = [-1.9, 1.9]
    theta_range = [-np.pi, np.pi]
    xdot_range = [-10, 10] #[-10, 10]
    thetadot_range = [-10, 10]

    grid_resolution = (51, 51, 51, 51)
    time_resolution = 101


    self.cartpole_deepreach = BaseCartPoleDeepreach(gravity=gravity, umax=umax, length=length, mass_cart=mass_cart, mass_pole=mass_pole,
                x_dist=x_dist, theta_dist=theta_dist, vel_dist=vel_dist, thetadot_dist=thetadot_dist, # disturbance bound parameters
                tMin=tMin, tMax=tMax)
    self.cartpole_hjr = BaseCartPoleHJR(self.cartpole_deepreach, gravity=gravity, umax=umax, length=length, mass_cart=mass_cart, mass_pole=mass_pole,
                x_dist=x_dist, theta_dist=theta_dist, vel_dist=vel_dist, thetadot_dist=thetadot_dist, # disturbance bound parameters
                tMin=tMin, tMax=tMax)
    

    # Initialize deepreach boundary function 
    self.cartpole_deepreach.init_boundary_fn(func=self.avoid_sdf)

    # HJ Reachability Solver Settings
    state_domain = hj.sets.Box(np.array([x_range[0], theta_range[0], xdot_range[0], thetadot_range[0]]), 
                               np.array([x_range[1], theta_range[1], xdot_range[1], thetadot_range[1]]))
    times = jnp.linspace(tMin, -tMax, time_resolution)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, 
                                                                   grid_resolution, 
                                                                   periodic_dims=1)
    
    
    # Setup Reach Avoid Problem 
    avoid_values = t2j(self.avoid_sdf(j2t(grid.states)))
    reach_values = t2j(self.reach_sdf(j2t(grid.states)))
    reach_avoid = lambda reach, avoid: (lambda t,v: jnp.minimum(jnp.maximum(v, reach), avoid)) # Reach avoid tube 
    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=reach_avoid(reach_values, avoid_values))

    initial_values = np.minimum(avoid_values, reach_values)

    # Solve HJI value function 
    if re_compute_hjr or not os.path.exists(hjr_filename): 
      all_values = hj.solve(solver_settings, self.cartpole_hjr, grid, times, initial_values, progress_bar=True)
      np.save(file=hjr_filename, arr=np.array(all_values))
    else: 
      all_values = np.load(file=hjr_filename)
      all_values = jnp.array(all_values)

    target_values = all_values[-1]
    diffs = -jnp.diff(all_values, axis=0).mean(axis=(1,2,3,4)) # 0 is time
    print("Final value function difference: ", diffs[-1])
    print("\n\n\n\n")

    # Create general safe environment attributes to be used externally 
    self.hjr_object = self.cartpole_hjr
    self.deepreach_object = self.cartpole_deepreach
    self.hjr_grid = grid 
    self.hjr_all_values = all_values 
    self.hjr_times = times 

    self.hjr_target_values = target_values 
    hjr_grid_interpolator_v = jax.vmap(self.hjr_grid.interpolate, in_axes=(None, 0))
    self.hjr_state_to_value = lambda state: hjr_grid_interpolator_v(jnp.array(self.hjr_all_values[-1]), state[None])[0]

    return 
  
  def obs_to_cbfstate(self, obs):
    if isinstance(obs, torch.Tensor):
      obs_cpu = obs.cpu().numpy()
    else: 
      obs_cpu = np.array(obs)

    x = obs_cpu[..., 0]
    theta = (np.arctan2(obs_cpu[..., 2], obs_cpu[..., 1]) + np.pi) % (2*np.pi) - np.pi
    xdot = obs_cpu[..., 3]
    thetadot = obs_cpu[..., 4]
    
    try:
      state = np.array([x[0], theta[0],xdot[0], thetadot[0]])
    except: 
      state = np.array([x, theta, xdot, thetadot])
    return state 
  
  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.

    Args:
      physics: An instance of `Physics`.
    """
    nv = physics.model.nv
    
    max_counter = 1000 
    counter = 0 
    found_start = False

    # NOTE: right now only support for one pole
    while not found_start: 
      if self.init_in_target:
        start_state = self.sample_state_in_target_set()
      else: 
        if self._swing_up: 
          x = self.random.uniform(-1, 1) #.01*self.random.randn()
          theta = np.pi + .01*self.random.randn()
        else: 
          x = self.random.uniform(-.1, .1)
          theta = self.random.uniform(-.034, .034, nv - 1)

        xdot = 0.01 * self.random.randn()
        thetadot = 0.01 * self.random.randn()
        start_state = np.array([x, theta, xdot, thetadot])

      start_state_value = self.hjr_state_to_value(start_state)

      if start_state_value >= 0:
        # safe 
        found_start = True
      else: 
        counter += 1
        if counter > max_counter: 
          # Force 0 and print that it occured
          start_state = np.array([0.0, np.pi, 0.0, 0.0])      
          start_state_val = self.hjr_state_to_value(start_state)
          print("\n\n\n\nMax counter exceeded: forcing to ", start_state)
          print("Start state value: ", start_state_val)
          print("\n\n\n")
          found_start = True 

    physics.named.data.qpos['slider'] = start_state[0]
    physics.named.data.qpos['hinge_1'] = start_state[1]
    physics.named.data.qvel[0] = start_state[2]
    physics.named.data.qvel[1] = start_state[3]

    # if self._swing_up:
    #   physics.named.data.qpos['slider'] = .01*self.random.randn()
    #   physics.named.data.qpos['hinge_1'] = np.pi + .01*self.random.randn()
    #   physics.named.data.qpos[2:] = .1*self.random.randn(nv - 2)
    # else:
    #   physics.named.data.qpos['slider'] = self.random.uniform(-.1, .1)
    #   physics.named.data.qpos[1:] = self.random.uniform(-.034, .034, nv - 1)
    # physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

    super().initialize_episode(physics)
    # base.Task.initialize_episode(self, physics)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()

    if self.is_unsafe(physics=physics): 
      physics.named.model.geom_rgba['pole_1'] = [1, 0, 0, 1] # force red for now
      physics.named.model.geom_rgba['cart'] = [1, 0, 0, 1] # force red for now
    elif self.reach_sdf(np.array([self.get_environment_state(physics=physics)], dtype=np.float32)) > 0: # In reach set
      physics.named.model.geom_rgba['pole_1'] = [0, 0, 1, 1] # force blue for now 
      physics.named.model.geom_rgba['cart'] = [0, 0, 1, 1] # force blue for now 
    else: 
      physics.named.model.geom_rgba['pole_1'] = [0.5, 0.5, 0.5, 1] # default back to beige
      physics.named.model.geom_rgba['cart'] = [0.5, 0.5, 0.5, 1] # default back to beige
    return obs

  def _get_reward(self, physics, sparse):
    if sparse:
      cart_in_bounds = rewards.tolerance(physics.cart_position(),
                                         self._CART_RANGE)
      angle_in_bounds = rewards.tolerance(physics.pole_angle_cosine(),
                                          self._ANGLE_COSINE_RANGE).prod()
      return cart_in_bounds * angle_in_bounds
    else:
      upright = (physics.pole_angle_cosine() + 1) / 2
      centered = rewards.tolerance(physics.cart_position(), margin=2)
      centered = (1 + centered) / 2
      small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic')[0]
      small_control = (4 + small_control) / 5
      small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
      small_velocity = (1 + small_velocity) / 2
      return upright.mean() * small_control * small_velocity * centered

  def get_reward(self, physics):
    """Returns a sparse or a smooth reward, as specified in the constructor."""
    return self._get_reward(physics, sparse=self._sparse)