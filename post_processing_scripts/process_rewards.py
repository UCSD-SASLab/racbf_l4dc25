"""
File to post process the rewards from the RL experiments. 
"""

import os 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 

current_folder = os.path.dirname(__file__)
custom_envs_path = os.path.join(current_folder, "../")
sys.path.append(custom_envs_path)

import custom_envs
import dmc2gym
import argparse 

"""
1. Load the rewards 
2. Compute the cumulative rewards - up to cutoff time 
3. Plot the cumulative rewards 
"""

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process a user-specified directory.")
    # parser.add_argument("-d", "--directory", type=str, help="The path to the directory to be used by the script.")
    # args = parser.parse_args()

    state_logs_dir = "" #args.directory #""
    # imgs_dir = "./images"
    imgs_dir = os.path.join(state_logs_dir, "./images")
    reward_tally_cutoff = 1000 # Cutoff time for the rewards

    rewards_file = os.path.join(state_logs_dir, "rewards.npy")

    # Load the rewards 
    rewards = np.array(np.load("rewards.npy"))
    print("Rewards Shape: ", rewards.shape)

    # Compute the cumulative rewards 
    cumulative_rewards = np.sum(rewards[:, :reward_tally_cutoff], axis=1)
    print("Cumulative Rewards Shape: ", cumulative_rewards.shape)

    # Plot the cumulative rewards 
    plt.figure()
    plt.plot(cumulative_rewards[0, :], label="Episode 1")
    plt.savefig(os.path.join(imgs_dir, "cumulative_rewards.png"))

    print("Done Processing Rewards!")