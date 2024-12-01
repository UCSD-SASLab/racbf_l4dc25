"""
File to post-process the final states and their delta from the target region
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from copy import deepcopy 

def write_data_to_file(x, mean, std_dev, filename, num_std=2): 
    """
    Write data to a file so that it can be used to generate tix plots later
    args: 
        - x: x values to write to file
        - mean: mean y values to write to file 
        - std_dev: std_dev y values to generate upper and lower bounds to write 
        - filename: filename to write the data to
        - num_std: number of standard deviations to use for the upper and lower bounds
    """
    with open(filename, "w") as f: 
        # Column Headers
        f.write("t x c1 c2\n")

        for i in range(len(x)): 
            c1 = float(mean[i] - num_std * std_dev[i])
            c2 = float(mean[i] + num_std * std_dev[i])

            f.write(f"{float(x[i]):.6e} {float(mean[i]):.6e} {c1:.6e} {c2:.6e}\n")
    return

def get_delta_from_target(final_states, abs_delta_x): 
    """
    args: 
        - final_states: (num episodes, final state dim)
        - abs_delta_x: if True, then take the absolute value of the delta x
    """
    # Define Target Region
    x_range = [-1.1, -0.8]
    not_target_theta_range = [-np.pi + 0.25, np.pi - 0.25] # If in this range then not in the target - other ranges describe target set
    xdot_range = [-0.1, 0.1]
    thetadot_range = [-0.25, 0.25]

    # Compute the deltas
    x = final_states[:, 0]
    x_delta = np.zeros_like(x)
    if abs_delta_x:
        x_delta[np.where(x<x_range[0])] = np.abs(x[np.where(x<x_range[0])] - x_range[0])
        x_delta[np.where(x>x_range[1])] = np.abs(x[np.where(x>x_range[1])] - x_range[1])
    else: 
        x_delta[np.where(x<x_range[0])] = x[np.where(x<x_range[0])] - x_range[0]
        x_delta[np.where(x>x_range[1])] = x[np.where(x>x_range[1])] - x_range[1]

    theta = final_states[:, 1]
    theta_delta = np.zeros_like(theta)
    theta_not_in_target_idx = np.where((theta > not_target_theta_range[0]) & (theta < not_target_theta_range[1]))
    theta_not_in_target = theta[theta_not_in_target_idx]
    theta_delta_not_in_target = np.zeros_like(theta_not_in_target)
    theta_delta_not_in_target[np.where(theta_not_in_target <= 0)] = theta_not_in_target[np.where(theta_not_in_target <= 0)] - not_target_theta_range[0]
    theta_delta_not_in_target[np.where(theta_not_in_target > 0)] = theta_not_in_target[np.where(theta_not_in_target > 0)] - not_target_theta_range[1]
    theta_delta[theta_not_in_target_idx] = theta_delta_not_in_target

    xdot = final_states[:, 2]
    xdot_delta = np.zeros_like(xdot)
    xdot_delta[np.where(xdot<xdot_range[0])] = xdot[np.where(xdot<xdot_range[0])] - xdot_range[0]
    xdot_delta[np.where(xdot>xdot_range[1])] = xdot[np.where(xdot>xdot_range[1])] - xdot_range[1]

    thetadot = final_states[:, 3]
    thetadot_delta = np.zeros_like(thetadot)
    thetadot_delta[np.where(thetadot<thetadot_range[0])] = thetadot[np.where(thetadot<thetadot_range[0])] - thetadot_range[0]
    thetadot_delta[np.where(thetadot>thetadot_range[1])] = thetadot[np.where(thetadot>thetadot_range[1])] - thetadot_range[1]

    delta_target_states = np.stack([x_delta, theta_delta, xdot_delta, thetadot_delta], axis=1)

    return delta_target_states

def get_sublist_mean_std(sublist_states):
    """
    args: 
        sublist_states: (num files in sublist, num episodes, final state dim)
    returns: 
        - mean: (num episodes, final state dim)
        - std_dev: (num episodes, final state dim)
    """
    # Compute the mean and standard deviation of each sublist
    mean = np.mean(sublist_states, axis=0)
    std_dev = np.std(sublist_states, axis=0)
    return mean, std_dev

def plot_combined_scatterplot(labels, all_means, all_stds, img_title, img_folder, delta_state, abs_delta_x, num_stds=2):
    """
    args: 
        - labels: list of labels for each sublist
        - all_means: list of mean states for each sublist 
        - all_stds: list of std dev states for each sublist
    """
    # Create a single figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(img_title, fontsize=16)

    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # List of colors for different datasets

    tix_folder = os.path.join(img_folder, "final_state_tix_files")
    os.makedirs(tix_folder, exist_ok=True)
    state_names = ["x", "theta", "xdot", "thetadot"]
    for idx in range(len(labels)):
        label = labels[idx]
        means = all_means[idx]
        stds = all_stds[idx]

        color = colors[idx % len(colors)]  # Cycle through colors if there are more datasets

        # Subplot 1: Time vs. State x
        axs[0, 0].plot(means[:, 0], label=label, color=color)
        axs[0, 0].fill_between(range(len(means)), means[:, 0] - num_stds * stds[:, 0], means[:, 0] + num_stds * stds[:, 0], color=color, alpha=0.2)
        axs[0, 0].set_xlabel("Episodes")
        if do_delta:
            if abs_delta_x:
                axs[0, 0].set_title("|Delta x| from Target")
                axs[0, 0].set_ylabel("|Delta x|")
            else: 
                axs[0, 0].set_title("Delta x from Target")
                axs[0, 0].set_ylabel("Delta x")
        else:
            axs[0, 0].set_title("State x")
            axs[0, 0].set_ylabel("x")
        axs[0, 0].legend()

        # Subplot 2: Time vs. State theta
        axs[0, 1].plot(means[:, 1], label=label, color=color)
        axs[0, 1].fill_between(range(len(means)), means[:, 1] - num_stds * stds[:, 1], means[:, 1] + num_stds * stds[:, 1], color=color, alpha=0.2)
        axs[0, 1].set_xlabel("Episodes")
        if do_delta:
            axs[0, 1].set_title("Delta Theta from Target")
            axs[0, 1].set_ylabel("Delta Theta")
        else: 
            axs[0, 1].set_title("State Theta")
            axs[0, 1].set_ylabel("Theta")
        axs[0, 1].legend()

        # Subplot 3: Time vs. State xdot
        axs[1, 0].plot(means[:, 2], label=label, color=color)
        axs[1, 0].fill_between(range(len(means)), means[:, 2] - num_stds * stds[:, 2], means[:, 2] + num_stds * stds[:, 2], color=color, alpha=0.2)
        axs[1, 0].set_xlabel("Episodes")
        if do_delta:
            axs[1, 0].set_title("Delta xdot from Target")
            axs[1, 0].set_ylabel("Delta xdot")
        else: 
            axs[1, 0].set_title("State xdot")
            axs[1, 0].set_ylabel("xdot")
        axs[1, 0].legend()

        # Subplot 4: Time vs. State thetadot
        axs[1, 1].plot(means[:, 3], label=label, color=color)
        axs[1, 1].fill_between(range(len(means)), means[:, 3] - num_stds * stds[:, 3], means[:, 3] + num_stds * stds[:, 3], color=color, alpha=0.2)
        axs[1, 1].set_xlabel("Episodes")
        if do_delta:
            axs[1, 1].set_title("Delta Thetadot from Target")
            axs[1, 1].set_ylabel("Delta Thetadot")
        else: 
            axs[1, 1].set_title("State Thetadot")
            axs[1, 1].set_ylabel("Thetadot")
        axs[1, 1].legend()

        # Save Tix files for each subplot
        for state_idx in range(4): 
            state_name = state_names[state_idx]
            tix_filename = os.path.join(tix_folder, f"{label}_{state_name}.txt")
            write_data_to_file(range(len(means)), means[:, state_idx], stds[:, state_idx], tix_filename, num_stds)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined figure
    if delta_state: 
        if abs_delta_x: 
            img_path = os.path.join(img_folder, "combined_final_states_scatterplot_delta_absdx.png")
        else: 
            img_path = os.path.join(img_folder, "combined_final_states_scatterplot_delta.png")
    else:
        img_path = os.path.join(img_folder, "combined_final_states_scatterplot_2.png")
    plt.savefig(img_path)
    print(f"Saved combined scatterplot at {img_path}")
    plt.show()


if __name__ == "__main__":
    do_delta = True 
    abs_delta_x = False
    num_stds = 1

    # base_dir = "/Users/nikhilushinde/Documents/Grad/research/arclab/RACBF_24/backup_results/all_state_logs_2024.11.29/"
    # directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
    #                ["0538_safeCartpoleAvoid_swingup_sacRACBF_swingup_43_5_resetTrue_spFalse", "2156_safeCartpoleAvoid_swingup_sacRACBF_swingup_53_5_resetTrue_spFalse"], #["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
    #                ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
    #                # NOTE: TODO: This last group is still incomplete! need to fix this later! 
    #                ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse"]]

    base_dir = "/Users/nikhilushinde/Documents/Grad/research/arclab/RACBF_24/backup_results/all_state_logs_2024.11.29/"
    directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
                   ["0538_safeCartpoleAvoid_swingup_sacRACBF_swingup_43_5_resetTrue_spFalse", "2156_safeCartpoleAvoid_swingup_sacRACBF_swingup_53_5_resetTrue_spFalse", "1251_safeCartpoleAvoid_swingup_sacRACBF_swingup_33_5_resetTrue_spFalse", "1801_safeCartpoleAvoid_swingup_sacRACBF_swingup_13_5_resetTrue_spFalse", "2023_safeCartpoleAvoid_swingup_sacRACBF_swingup_23_5_resetTrue_spFalse"], #["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
                   ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
                   # NOTE: TODO: This last group is still incomplete! need to fix this later! 
                   ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse", ]]

    for sub_dir_num in range(len(directories)):
        for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
            directories[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), "state_logs")
            
    labels = ["sac-racbf", 
              "sac-cbf", 
              "sac", 
              "sac-racbf-noreset"]

    # Create images folder
    imgs_dir = "./images"
    os.makedirs(imgs_dir, exist_ok=True)

    sdf_values = {}
    final_states_dict = {}

    # Process each directory
    all_mean_sublist_states = []
    all_std_dev_sublist_states = []

    for i in range(len(labels)): 

        directory_list = directories[i]
        curr_sublist_states = []

        for dir_num, directory in enumerate(directory_list): 
            label = str(labels[i]) + "-" + str(dir_num)
            print(f"Processing directory: {directory} with label: {label}")

            final_states_file = os.path.join(directory, "final_states.npy")
            all_states_file = os.path.join(directory, "states.npy")

            if not os.path.isfile(final_states_file) or not os.path.isfile(all_states_file):
                print(f"Warning: Missing required files in {directory}. Skipping...")
                continue

            # Load the states
            final_states = np.load(final_states_file, allow_pickle=True)
            if do_delta: 
                normal_final_states = deepcopy(final_states)
                final_states = get_delta_from_target(final_states, abs_delta_x=abs_delta_x)
            
            # if i >= 2: 
            #     print(label)
            #     import pdb; pdb.set_trace()

            # all_states = np.load(all_states_file)

            # # Do assertion check to verify
            # all_final_states = np.array(all_states)[:, -1, :]
            # assert np.allclose(final_states, all_final_states), (
            #     "Final states are not correct. Check the final states and all states."
            # )

            # Store the data for combined plotting
            curr_sublist_states.append(final_states)

        curr_mean_sublist_states, curr_std_dev_sublist_states = get_sublist_mean_std(np.array(curr_sublist_states))
        all_mean_sublist_states.append(curr_mean_sublist_states)
        all_std_dev_sublist_states.append(curr_std_dev_sublist_states)

        # print(labels[i])
        # import pdb; pdb.set_trace()
            

    # Plot the combined scatterplot of the final states
    plot_combined_scatterplot(
        labels=labels, 
        all_means=np.array(all_mean_sublist_states), 
        all_stds=np.array(all_std_dev_sublist_states),
        img_title="Combined Final States Scatterplot",
        img_folder=imgs_dir, 
        delta_state=do_delta,
        abs_delta_x=abs_delta_x,
        num_stds=num_stds
    )
