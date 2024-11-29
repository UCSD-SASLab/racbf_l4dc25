"""
SAC Agent with a CBF Safety Filter
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import utils

import hydra


class SACCBFAgent(Agent):
    """SAC CBF algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, 
                 # 
                 # Safety Related Parameters
                 deepreach_object, hjr_object, hjr_grid, hjr_all_values, hjr_times, 
                 obs_to_cbfstate, 
                 cbf_alpha_value):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

        # NOTE: this is only if it is the inverted pendulum environment for now 
        ############ START: CBF Safety Filter ##############
        # import sys 
        # import os 
        # deepreach_path = os.path.join(os.getcwd(), "../../../../../")
        # print(os.listdir(os.path.join(os.getcwd(),"../../../../../")))
        # print(os.path.join(os.getcwd(),"../../../../../"))
        # assert(os.path.exists(deepreach_path))
        # sys.path.append(deepreach_path)
        # import hj_reachability as hj
        # from deepreach.dynamics.dynamics import InvertedPendulum
        # from deepreach.dynamics.dynamics_hjr import InvertedPendulum as InvertedPendulumHJR
        # from cbf_opt_kit.cbf import HJReachabilityControlAffineCBF
        # from cbf_opt_kit.safety_filter import ControlAffineSafetyFilter

        # from torch2jax import t2j, j2t
        # import jax 
        # import jax.numpy as jnp

        # gravity = 9.81
        # length = 1.0
        # mass = 1.0 

        # unsafe_theta_min = np.pi/2 #np.pi/2
        # unsafe_theta_max = np.pi/2 + np.pi/16 #3*np.pi/4
        # min_torque = -1.0
        # max_torque = 1.0 

        # max_theta_dist = 0.01
        # max_thetadot_dist = 0 
        # tMin = 0.0
        # tMax = 10

        # theta_range = [-np.pi, np.pi]
        # thetadot_range = [-10, 10]
        # grid_resolution = (201, 101)
        # time_resolution = 101

        # alpha_value = 0.1

        # # Create inverted pendulum
        # self.inv_pendulum = InvertedPendulum(gravity=gravity, 
        #                                      length=length, 
        #                                      mass=mass, 
        #                                      unsafe_theta_min=unsafe_theta_min,
        #                                      unsafe_theta_max=unsafe_theta_max, 
        #                                      min_torque=min_torque, 
        #                                      max_torque=max_torque,
        #                                      max_theta_dist=max_theta_dist, 
        #                                      max_thetadot_dist=max_thetadot_dist,  
        #                                      tMin=tMin, 
        #                                      tMax=tMax)
        # self.inv_pendulum_hjr = InvertedPendulumHJR(torch_dynamics=self.inv_pendulum, 
        #                                             gravity=gravity, 
        #                                             length=length,
        #                                             mass=mass, 
        #                                             unsafe_theta_min=unsafe_theta_min, 
        #                                             unsafe_theta_max=unsafe_theta_max, 
        #                                             min_torque=min_torque, 
        #                                             max_torque=max_torque,
        #                                             max_theta_dist=max_theta_dist, 
        #                                             max_thetadot_dist=max_thetadot_dist, 
        #                                             tMin=tMin, 
        #                                             tMax=tMax)

        # # HJ Reachability Solver Settings
        # state_domain = hj.sets.Box(np.array([theta_range[0], thetadot_range[0]]), 
        #                            np.array([theta_range[1], thetadot_range[1]]))
        # grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, 
        #                                                                     grid_resolution, 
        #                                                                     periodic_dims=0)
        # sdf_values = t2j(self.inv_pendulum_hjr.torch_dynamics.boundary_fn(j2t(grid.states)))
        # times = jnp.linspace(tMin, -tMax, time_resolution)
        # initial_values = sdf_values 
        # solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

        # # Solve HJI value function
        # all_values = hj.solve(solver_settings, self.inv_pendulum_hjr, grid, times, initial_values, progress_bar=True)
        # target_values = all_values[-1]
        # diffs = -jnp.diff(all_values, axis=0).mean(axis=(1,2))
        # desired_diff_epsilon = 1e-10
        # assert(diffs[-1] < desired_diff_epsilon)

        from cbf_opt_kit.cbf import HJReachabilityControlAffineCBF
        from cbf_opt_kit.safety_filter import ControlAffineSafetyFilter

        # Create CBF Safety Filter
        class ReachabilityModel:
            def __init__(self, grid, grid_values, times): 
                self.grid = grid
                self.grid_values = grid_values
                self.times = times 
                return 
        
        self.deepreach_object = deepreach_object
        self.hjr_object = hjr_object
        self.hjr_grid = hjr_grid
        self.hjr_all_values = hjr_all_values
        self.hjr_times = hjr_times 
        self.obs_to_cbfstate = obs_to_cbfstate
        self.cbf_alpha_value = cbf_alpha_value

        model = ReachabilityModel(grid=self.hjr_grid, grid_values=self.hjr_all_values[-1], times=self.hjr_times)
        cbf = HJReachabilityControlAffineCBF(dynamics=self.hjr_object, model=model, time_invariant=True)
        safety_filter = ControlAffineSafetyFilter(cbf, alpha = lambda x: self.cbf_alpha_value * x, limit_controls=True)
        safety_filter.dynamics.control_dim = 1

        self.target_time = self.hjr_times[-1]
        self.safety_filter = safety_filter

        ############ END: CBF Safety Filter ##############

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)

        # NOTE: this is only if it is the inverted pendulum environment for now 
        ################# START: CBF Safety Filter ################
        action_np = action.detach().cpu().numpy()
        scaled_up_action = action_np * self.hjr_object.umax # NOTE: scale up action from -1 to 1 to hjr range
        cbf_state = self.obs_to_cbfstate(obs=obs)
        cbf_action, d, boolean = self.safety_filter(state=cbf_state, time=self.target_time, nominal_control=scaled_up_action)
        cbf_action = cbf_action / self.hjr_object.umax # NOTE: scale down action from hjr range to -1 to 1
        cbf_action = np.array(cbf_action, dtype=np.float32)
        cbf_action_delta = cbf_action - action_np 
        action = action + torch.from_numpy(cbf_action_delta).to(action.device).reshape(action.shape)

        action = action.clamp(*self.action_range).float()
        ################# END: CBF Safety Filter ################

        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
