"""
Train file for training with safe environments: 
Main differences: 
    - environments: have hj reachability based attributes to enable safe initialization
    - agents: perform actions with safety filters using the environment's hj reachability attributes. 
"""
#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import dmc2gym
import hydra

# Custom Safe Environments
import custom_envs 
from custom_envs import safePendulum 


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


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        ############ START: CBF Safety Specific ############
        # Assign safety configuration parameters
        # NOTE: cfg wants a primitive so doing the below approach
        # cfg.deepreach_object = self.env.task.deepreach_object
        # cfg.hjr_object = self.env.task.hjr_object
        # cfg.hjr_grid = self.env.task.hjr_grid
        # cfg.hjr_all_values = self.env.task.hjr_all_values
        # cfg.hjr_times = self.env.task.hjr_times
        # cfg.obs_to_cbf_state = self.env.task.obs_to_cbf_state

        agent_class = hydra.utils.get_class(cfg.agent["class"])
        self.agent = agent_class(obs_dim=cfg.agent.params.obs_dim, 
                                 action_dim=cfg.agent.params.action_dim, 
                                 action_range=cfg.agent.params.action_range, 
                                 device=cfg.agent.params.device, 
                                 critic_cfg=cfg.agent.params.critic_cfg, 
                                 actor_cfg=cfg.agent.params.actor_cfg, 
                                 discount=cfg.agent.params.discount, 
                                 init_temperature=cfg.agent.params.init_temperature, 
                                 alpha_lr=cfg.agent.params.alpha_lr, 
                                 alpha_betas=cfg.agent.params.alpha_betas, 
                                 actor_lr=cfg.agent.params.actor_lr, 
                                 actor_betas=cfg.agent.params.actor_betas,
                                 actor_update_frequency=cfg.agent.params.actor_update_frequency, 
                                 critic_lr=cfg.agent.params.critic_lr, 
                                 critic_betas=cfg.agent.params.critic_betas, 
                                 critic_tau=cfg.agent.params.critic_tau, 
                                 critic_target_update_frequency=cfg.agent.params.critic_target_update_frequency, 
                                 batch_size=cfg.agent.params.batch_size, 
                                 learnable_temperature=cfg.agent.params.learnable_temperature, 
                                 deepreach_object=self.env.task.deepreach_object, 
                                 hjr_object=self.env.task.hjr_object, 
                                 hjr_grid=self.env.task.hjr_grid, 
                                 hjr_all_values=self.env.task.hjr_all_values, 
                                 hjr_times=self.env.task.hjr_times, 
                                 obs_to_cbfstate=self.env.task.obs_to_cbfstate, 
                                 cbf_alpha_value=cfg.agent.params.cbf_alpha_value)

        # self.agent = hydra.utils.instantiate(cfg.agent, deepreach_object=self.env.task.deepreach_object, hjr_object=self.env.task.hjr_object, hjr_grid=self.env.task.hjr_grid, hjr_all_values=self.env.task.hjr_all_values, hjr_times=self.env.task.hjr_times, obs_to_cbf_stat=self.env.task.obs_to_cbf_state)
        ############ END: CBF Safety Specific ############

        # self.agent = hydra.utils.instantiate(cfg.agent, hjr_dict)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                # obs, reward, done, _ = self.env.step(action)
                obs, reward, done, _, _ = self.env.step(action) # NOTE: Nikhil Add 
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            # next_obs, reward, done, _ = self.env.step(action)
            next_obs, reward, done, _, _ = self.env.step(action) # NOTE: NIKHIL ADD 

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train_safe.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
