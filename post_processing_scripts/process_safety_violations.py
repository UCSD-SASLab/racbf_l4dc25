"""
File to create a plot of safety violations from the RL experiments. 
"""

import numpy as np
import os 
import sys 

current_folder = os.path.dirname(__file__)
custom_envs_path = os.path.join(current_folder, "../")
sys.path.append(custom_envs_path)

import custom_envs
import dmc2gym
import argparse

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a user-specified directory.")
    parser.add_argument("-d", "--directory", type=str, help="The path to the directory to be used by the script.")
    args = parser.parse_args()

    state_logs_dir = args.directory # ""

    # Load the safety violations from the RL experiments
    safety_violations = np.load("safety_violations.npy")
    safety_violations = np.sum(safety_violations)

    # Plot the safety violations 
    print("Num safety violations", safety_violations)
