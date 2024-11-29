"""
File to post process the rewards from the RL experiments. 
"""
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 

current_folder = os.path.dirname(__file__)
custom_envs_path = os.path.join(current_folder, "../")
sys.path.append(custom_envs_path)

import custom_envs 
import dmc2gym 
import argparse 

"""
1. Load the environment 
2. Load the final states 
    2.1: Load all states and verify that the final states are correct/as you expect 
3. Compute sdf for each of the final states 
    3.1: Plot the sdf values for the final states for each episode 
4. Create a scatterpolot of the final states - x, theta, xdot, thetadot 
"""

def plot_final_states_scatterplot(states, img_title, img_folder):
    # Create a single figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(img_title, fontsize=16)

    # Subplot 1: Time vs. State x
    axs[0, 0].plot(states[:, 0], label="states", color='g')
    axs[0, 0].set_title("State x")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("x")
    axs[0, 0].legend()

    # Subplot 2: Time vs. State theta
    axs[0, 1].plot(states[:, 1], label="states", color='g')
    axs[0, 1].set_title("State Theta")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Theta")
    axs[0, 1].legend()

    # Subplot 3: Time vs. State xdot
    axs[1, 0].plot(states[:, 2], label="states", color='g')
    axs[1, 0].set_title("State xdot")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("xdot")
    axs[1, 0].legend()

    # Subplot 4: Time vs. State thetadot
    axs[1, 1].plot(states[:, 3], label="states", color='g')
    axs[1, 1].set_title("State Thetadot")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Thetadot")
    axs[1, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined figure
    plt.savefig(os.path.join(img_folder, "final_states_cluster.png"))
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a user-specified directory.")
    parser.add_argument("-d", "--directory", type=str, help="The path to the directory to be used by the script.")
    args = parser.parse_args()

    state_logs_dir = args.directory #"" # path to state logs 
    imgs_dir = "./images"
    # imgs_dir = os.path.join(state_logs_dir, "./images/")

    os.makedirs(imgs_dir, exist_ok=True)

    final_states_file = os.path.join(state_logs_dir, "final_states.npy")
    all_states_file = os.path.join(state_logs_dir, "states.npy")

    # Load environment: primary for access to reach sdf 
    env = dmc2gym.make(domain_name="safeCartpoleRA", task_name="swingup")
    env.reset() 

    # Load the states 
    final_states = np.load(final_states_file)
    all_states = np.load(all_states_file)
    final_states_sdf_vals = env.task.reach_sdf(np.array(final_states, dtype=np.float32))

    # Do assertion check to verify 
    all_final_states = np.array(all_states)[:, -1, :] 
    assert np.allclose(final_states, all_final_states), "Final states are not correct. Check the final states and all states"

    # Plot the sdf values for the final states
    final_sdf_name = "final_states_sdf_vals.png"
    plt.figure() 
    plt.plot(final_states_sdf_vals)
    plt.xlabel("Episode")
    plt.ylabel("SDF Value")
    plt.title("SDF Values for Final States")
    plt.savefig(os.path.join(imgs_dir, final_sdf_name))
    print("Plotted the sdf values")

    # Create scatterplot of the final states
    plot_final_states_scatterplot(states=all_final_states, img_title="Final States Scatterplot",  img_folder=imgs_dir)
    print("Plotted the final states scatterplot")