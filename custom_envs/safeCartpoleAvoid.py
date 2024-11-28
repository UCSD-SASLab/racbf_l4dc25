"""
Safe cartpole environment for sacCBF agent with just the avoid hjr solution
Should inherit everything from safeCartpoleRA.py to ensure consistency between environments
"""

try: 
  # When running inside module
  from utils import BaseCartPoleDeepreach, BaseCartPoleHJR
  from safeCartpoleRA import BalanceRA 
  from safeCartpoleRA import get_model_and_assets, _make_model, _DEFAULT_TIME_LIMIT, CURR_FILE_PATH, Physics
except: 
  # When running from outside module
  from custom_envs.utils import BaseCartPoleDeepreach, BaseCartPoleHJR
  from custom_envs.safeCartpoleRA import BalanceRA
  from custom_envs.safeCartpoleRA import get_model_and_assets, _make_model, _DEFAULT_TIME_LIMIT, CURR_FILE_PATH, Physics


from dm_control.rl import control
from dm_control.utils import containers
import os 

import hj_reachability as hj
from torch2jax import t2j, j2t
import jax 
import jax.numpy as jnp 
import numpy as np 

SUITE = containers.TaggedTasks()

"""
safeCartpoleAvoid domain
"""

@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = BalanceAvoid(swing_up=True, sparse=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add('benchmarking')
def swingup_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the sparse reward variant of the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = BalanceAvoid(swing_up=True, sparse=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class BalanceAvoid(BalanceRA): 
  def setup_hj_reachability(self): 
    # Override setup_hj_reachability to only generate avoid solution
    # NOTE: ensure parameters are the same as BalanceRA for consistency 

    # Unsafe Sliver Configuration
    # For debugging purposes
    re_compute_hjr = False #True #True # False
    hjr_filename = "rl_safeCartpoleAvoid_hjr_values.npy"
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
    solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    initial_values = avoid_values 

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