"""
Train Safe with Reach Avoid: 

What to do: 
1. Create sacRACBF agent 
2. Have global variable keeping track of last state of environment - on training iteration and evaluation iteration 
3. Every time you reset the environment force set the environment to the last state of the training iteration. 
4. SAC agent act() takes in (obs, time and sample)

TODO: NOTE: later will have to modify the reward logging because right now it will continue to log the reward in the return to start phase 
"""


"""
Train file for training with safe environments with reach avoid agents: 
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
import shutil 

# Custom Safe Environments
import custom_envs 
from custom_envs import safePendulum, safeCartpole, safeCartpoleRA


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

def reset_environment(env):
    """
    Reset the environment back to the current state of the environment. 
    args: 
        - env: environment object to reset 
    returns: 
        - env: environment object after reset 
    """
    state_to_reset = env.task.get_environment_state(physics=env.physics)
    reset_obs = env.reset()
    updated_reset_obs = env.task.set_environment_state(physics=env.physics, state=state_to_reset)

    print("\n")
    print("State to reset: ", state_to_reset)
    print(("State it was reset to: ", env.task.get_environment_state(physics=env.physics)))

    return env, updated_reset_obs 

def reached_target(env): 
    # True if reached the target at the end of episode
    return env.task.reach_sdf(np.array([env.task.get_environment_state(physics=env.physics)], dtype=np.float32)) > 0

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
        self.env.reset() # start with reset 
        self.force_reset_env = cfg.force_reset_env

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        ############ START: RACBF Safety Specific ############

        # Environment Time attributes
        self.current_t = self.env.task.max_time
        self.dt = self.env.task.dt

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

        ############ END: CBF Safety Specific ############

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, height=480, width=480)
        self.step = 0

        # Setup Logging 
        self.states = []
        self.times = []
        self.actions = []
        self.rewards = []
        self.safety_violations = []

        self.final_states = []
        self.reached_target = []
        
    def save_state_logs(self): 
        # Save state logs 
        print("Saving state logs")
        state_log_path = os.path.join(self.work_dir, "state_logs")
        os.makedirs(state_log_path, exist_ok=True)

        np.save(os.path.join(state_log_path, "states.npy"), np.array(self.states), allow_pickle=True)
        np.save(os.path.join(state_log_path, "times.npy"), np.array(self.times), allow_pickle=True)
        np.save(os.path.join(state_log_path, "actions.npy"), np.array(self.actions), allow_pickle=True)
        np.save(os.path.join(state_log_path, "rewards.npy"), np.array(self.rewards), allow_pickle=True)
        np.save(os.path.join(state_log_path, "safety_violations.npy"), np.array(self.safety_violations), allow_pickle=True)
        np.save(os.path.join(state_log_path, "reached_target.npy"), np.array(self.reached_target), allow_pickle=True)
        np.save(os.path.join(state_log_path, "final_states.npy"), np.array(self.final_states), allow_pickle=True)

        return 

    def evaluate(self):
        print("\n\nStarting Evaluate Script")
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):

            if self.force_reset_env: 
                obs = self.env.reset()
            else:
                self.env, obs = reset_environment(self.env) # reset to last episode end state 
            # self.agent.reset()
            self.current_t = self.env.task.max_time 

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0

            # Setup Loggers
            curr_episode_states = []
            curr_episode_times = []
            curr_episode_actions = []
            curr_episode_rewards = []
            curr_episode_safety_violations = []

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, time=self.current_t, sample=False)
                
                # obs, reward, done, _ = self.env.step(action)
                obs, reward, done, _, _ = self.env.step(action) # NOTE: Nikhil Add 
                self.current_t -= self.dt

                # Update state log: 
                log_state = self.env.task.get_environment_state(physics=self.env.physics)
                curr_episode_states.append(log_state)
                curr_episode_times.append(self.current_t)
                curr_episode_actions.append(action)
                curr_episode_rewards.append(reward)
                curr_episode_safety_violations.append(self.env.task.is_unsafe(physics=self.env.physics))
                if done: 
                    self.reached_target.append(reached_target(env=self.env))
                    self.final_states.append(self.env.task.get_environment_state(physics=self.env.physics))

                    # Append episode logs 
                    self.states.append(curr_episode_states)
                    self.times.append(curr_episode_times)
                    self.actions.append(curr_episode_actions)
                    self.rewards.append(curr_episode_rewards)
                    self.safety_violations.append(curr_episode_safety_violations)

                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        # save the models
        saved_actor_pth, saved_critic_pth = utils.save_agent(base_dir=self.work_dir, sac_agent=self.agent, step_num=self.step)
        _, latest_actor_pth, latest_critic_pth = utils.get_model_paths(base_dir=self.work_dir, step_num=None)
        shutil.copy(saved_actor_pth, latest_actor_pth)
        shutil.copy(saved_critic_pth, latest_critic_pth)

        self.save_state_logs()

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        
        # Setup Loggers
        curr_episode_states = []
        curr_episode_times = []
        curr_episode_actions = []
        curr_episode_rewards = []
        curr_episode_safety_violations = []

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

                print("Resetting in training: ")
                if self.force_reset_env: 
                    obs = self.env.reset()
                else: 
                    self.env, obs = reset_environment(self.env) # reset to last episode end state
                # self.agent.reset()
                self.current_t = self.env.task.max_time

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                # Setup Loggers
                curr_episode_states = []
                curr_episode_times = []
                curr_episode_actions = []
                curr_episode_rewards = []
                curr_episode_safety_violations = []

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, time=self.current_t, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            # next_obs, reward, done, _ = self.env.step(action)
            next_obs, reward, done, _, _ = self.env.step(action) # NOTE: NIKHIL ADD 
            self.current_t -= self.dt

            # Update state log: 
            log_state = self.env.task.get_environment_state(physics=self.env.physics)
            curr_episode_states.append(log_state)
            curr_episode_times.append(self.current_t)
            curr_episode_actions.append(action)
            curr_episode_rewards.append(reward)
            curr_episode_safety_violations.append(self.env.task.is_unsafe(physics=self.env.physics))
            if done: 
                self.reached_target.append(reached_target(env=self.env))
                self.final_states.append(self.env.task.get_environment_state(physics=self.env.physics))

                # Append episode logs 
                self.states.append(curr_episode_states)
                self.times.append(curr_episode_times)
                self.actions.append(curr_episode_actions)
                self.rewards.append(curr_episode_rewards)
                self.safety_violations.append(curr_episode_safety_violations)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train_safe_RA.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
