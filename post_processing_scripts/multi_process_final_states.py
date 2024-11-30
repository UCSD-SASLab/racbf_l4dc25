"""
File to post-process the rewards from the RL experiments with multiple directories and labels.
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


def plot_combined_scatterplot(all_states_dict, img_title, img_folder):
    # Create a single figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(img_title, fontsize=16)

    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # List of colors for different datasets

    for idx, (label, states) in enumerate(all_states_dict.items()):
        color = colors[idx % len(colors)]  # Cycle through colors if there are more datasets

        # Subplot 1: Time vs. State x
        axs[0, 0].plot(states[:, 0], label=label, color=color)
        axs[0, 0].set_title("State x")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("x")
        axs[0, 0].legend()

        # Subplot 2: Time vs. State theta
        axs[0, 1].plot(states[:, 1], label=label, color=color)
        axs[0, 1].set_title("State Theta")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].set_ylabel("Theta")
        axs[0, 1].legend()

        # Subplot 3: Time vs. State xdot
        axs[1, 0].plot(states[:, 2], label=label, color=color)
        axs[1, 0].set_title("State xdot")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("xdot")
        axs[1, 0].legend()

        # Subplot 4: Time vs. State thetadot
        axs[1, 1].plot(states[:, 3], label=label, color=color)
        axs[1, 1].set_title("State Thetadot")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel("Thetadot")
        axs[1, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined figure
    img_path = os.path.join(img_folder, "combined_final_states_scatterplot.png")
    plt.savefig(img_path)
    print(f"Saved combined scatterplot at {img_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process multiple directories and plot combined results.")
    # parser.add_argument(
    #     "-d", "--directories", nargs="+", type=str, required=True,
    #     help="List of paths to the directories to be used by the script."
    # )
    # parser.add_argument(
    #     "-l", "--labels", nargs="+", type=str, required=True,
    #     help="List of labels corresponding to the directories for plotting."
    # )
    # args = parser.parse_args()

    # if len(args.directories) != len(args.labels):
    #     raise ValueError("The number of directories and labels must match.")

    # directories = args.directories
    # labels = args.labels

    # base_dir = "../exp/2024.11.28"
    # directories = ["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetTrue",
    #                "1010_safeCartpoleAvoid_swingup_sacCBF_safeswingup_14_12", 
    #                "1010_safeCartpoleRA_swingup_sac_test_exp_14", 
    #                "1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetFalse"]

    base_dir = "../exp/2024.11.29"
    directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
                   ["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
                   ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
                   # NOTE: TODO: This last group is still incomplete! need to fix this later! 
                   ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse"]]
    

    for sub_dir_num in range(len(directories)):
        for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
            directories[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), "state_logs")
            
    labels = ["sac-racbf-14", 
              "sac-cbf-14", 
              "sac-14", 
              "sac-racbf-noreset"]

    # Create images folder
    imgs_dir = "./images"
    os.makedirs(imgs_dir, exist_ok=True)

    # Load environment: primarily for access to reach SDF
    env = dmc2gym.make(domain_name="safeCartpoleRA", task_name="swingup")
    env.reset()

    sdf_values = {}
    final_states_dict = {}

    # Process each directory
    for i in range(len(labels)): 
        directory_list = directories[i]
        for dir_num, directory in enumerate(directory_list): 
            label = str(labels[i]) + "-" + str(dir_num)
            print(f"Processing directory: {directory} with label: {label}")

            final_states_file = os.path.join(directory, "final_states.npy")
            all_states_file = os.path.join(directory, "states.npy")

            if not os.path.isfile(final_states_file) or not os.path.isfile(all_states_file):
                print(f"Warning: Missing required files in {directory}. Skipping...")
                continue

            # Load the states
            final_states = np.load(final_states_file)
            all_states = np.load(all_states_file)
            final_states_sdf_vals = env.task.reach_sdf(np.array(final_states, dtype=np.float32))

            # Do assertion check to verify
            all_final_states = np.array(all_states)[:, -1, :]
            assert np.allclose(final_states, all_final_states), (
                "Final states are not correct. Check the final states and all states."
            )

            # Store the data for combined plotting
            sdf_values[label] = final_states_sdf_vals
            final_states_dict[label] = all_final_states

    # Plot the combined SDF values
    plt.figure()
    for label, sdf_vals in sdf_values.items():
        plt.plot(sdf_vals, label=label)
    plt.xlabel("Episode")
    plt.ylabel("SDF Value")
    plt.title("Combined SDF Values for Final States")
    plt.legend()
    sdf_plot_path = os.path.join(imgs_dir, "combined_final_states_sdf_vals.png")
    plt.savefig(sdf_plot_path)
    print(f"Saved combined SDF plot at {sdf_plot_path}")

    # Plot the combined scatterplot of the final states
    plot_combined_scatterplot(
        all_states_dict=final_states_dict,
        img_title="Combined Final States Scatterplot",
        img_folder=imgs_dir
    )
