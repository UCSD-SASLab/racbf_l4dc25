import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math

import dmc2gym


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


################## NIKHIL ADDED ##################

def get_model_paths(base_dir, step_num=None): 
    model_save_folder = os.path.join(base_dir, "models")
    if step_num is None: 
        actor_pth = os.path.join(model_save_folder, "actor.pth")
        critic_pth = os.path.join(model_save_folder, "critic.pth")
    else: 
        actor_pth = os.path.join(model_save_folder, f"actor_{step_num}.pth")
        critic_pth = os.path.join(model_save_folder, f"critic_{step_num}.pth")
    return model_save_folder, actor_pth, critic_pth

def save_agent(base_dir, sac_agent, step_num=None):
    """
    args: 
        - base_dir: the experiment directory where all the results of the experiment are located
        - sac_agent: the sac agent to save
    returns: 
        - actor_pth
        - critic_pth
    """
    model_save_folder, actor_pth, critic_pth = get_model_paths(base_dir, step_num=step_num)
    os.makedirs(model_save_folder, exist_ok=True)

    torch.save(sac_agent.actor.state_dict(), actor_pth)
    torch.save(sac_agent.critic.state_dict(), critic_pth)
    print("Saving actor to: ", actor_pth)
    print("Saving critic to: ", critic_pth)

    return actor_pth, critic_pth

def load_agent(base_dir, sac_agent, step_num=None): 
    """
    args: 
        - base_dir: the experiment directory where all the results of the experiment were located
    returns: 
        - sac_agent: agent with loaded actor and critic networks
    """
    model_save_folder, actor_pth, critic_pth = get_model_paths(base_dir, step_num)
    sac_agent.actor.load_state_dict(torch.load(actor_pth))
    sac_agent.critic.load_state_dict(torch.load(critic_pth))

    return sac_agent

from omegaconf import OmegaConf
import hydra 
def create_and_load_sac_agent(experiment_dir, env): 
    """
    Given an experiment folder: 
    1. Create a sac agent with the config file 
    2. load the actor and critic models in the sac agent 
    3. return the sac agent for use
    args: 
        - experiment_dir: the experiment directory
        - env: environment - necessary to complete configuration for sac agent
    """
    config_file = os.path.join(experiment_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(config_file)

    cfg.agent.params.obs_dim = env.observation_space.shape[0]
    cfg.agent.params.action_dim = env.action_space.shape[0]
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()), 
        float(env.action_space.high.max())
    ]

    agent = hydra.utils.instantiate(cfg.agent)
    agent = load_agent(base_dir=experiment_dir, sac_agent=agent)

    return agent 
