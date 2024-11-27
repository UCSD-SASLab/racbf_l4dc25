"""
File for Reach Avoid SAC Agent 
What to do: 
Implement SAC agent like in sacCBF: with following modifications - just inherit sacCBF and modify the following: 

1. init: Change init function to create a time invariant and time varying CBF 
2. act: Create a new act function 
"""

from agent.sac import SACAgent
import torch 
import hydra 
import utils 

class SACCBFRAAgent(SACAgent):
    """SAC RACBF algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, 
                 # Safety Related Parameters
                 deepreach_object, hjr_object, hjr_grid, hjr_all_values, hjr_times, 
                 obs_to_cbfstate, 
                 cbf_alpha_value):
        
        super().__init__(obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature)

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

        time_invariant_model = ReachabilityModel(grid=self.hjr_grid, grid_values=self.hjr_all_values[-1], times=self.hjr_times)
        time_varying_model = ReachabilityModel(grid=self.hjr_grid, grid_values=self.hjr_all_values, times=self.hjr_times)

        self.cbf_TI = HJReachabilityControlAffineCBF(dynamics=self.hjr_object, model=time_invariant_model, time_invariant=True) # Time Invariant 
        self.cbf_TV = HJReachabilityControlAffineCBF(dynamics=self.hjr_object, model=time_varying_model, time_invariant=False) # Time Varying

        self.safety_filter_TI = ControlAffineSafetyFilter(self.cbf_TI, alpha = lambda x: self.cbf_alpha_value * x, limit_controls=True)
        self.safety_filter_TV = ControlAffineSafetyFilter(self.cbf_TV, alpha = lambda x: self.cbf_alpha_value * x, limit_controls=True)
        self.safety_filter_TI.dynamics.control_dim = 1
        self.safety_filter_TV.dynamics.control_dim = 1

        self.target_time = self.hjr_times[-1]

        ############ END: CBF Safety Filter ##############
    
    def act(self, obs, time, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        action_np = utils.to_np(action[0])

        if time >= self.target_time: 
            # Time Invariant CBF 
            cbf = self.cbf_TI 
            safety_filter = self.safety_filter_TI
        else: 
            # Time Varying CBF
            cbf = self.cbf_TV
            safety_filter = self.safety_filter_TV
        
        # CBF Safety Filter 
        cbf_state = self.obs_to_cbfstate(obs=obs)
        cbf_action, d, boolean = safety_filter(state=cbf_state, time=self.target_time, nominal_control=action_np)
        cbf_action_delta = cbf_action - action_np 
        action = action + torch.from_numpy(cbf_action_delta).to(action.device).reshape(action.shape)

        action = action.clamp(*self.action_range).float()
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])