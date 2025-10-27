import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 16, "font.family": "serif", "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16, "figure.titlesize": 16, "lines.linewidth": 4})

# def write_data_to_file(x, mean, std_dev, filename, num_std=2): 
#     """
#     Write data to a file so that it can be used to generate tix plots later
#     args: 
#         - x: x values to write to file
#         - mean: mean y values to write to file 
#         - std_dev: std_dev y values to generate upper and lower bounds to write 
#         - filename: filename to write the data to
#         - num_std: number of standard deviations to use for the upper and lower bounds
#     """
#     with open(filename, "w") as f: 
#         # Column Headers
#         f.write("t x c1 c2\n")

#         for i in range(len(x)): 
#             c1 = float(mean[i] - num_std * std_dev[i])
#             c2 = float(mean[i] + num_std * std_dev[i])

#             f.write(f"{float(x[i]):.6e} {float(mean[i]):.6e} {c1:.6e} {c2:.6e}\n")
#     return

def plot_state_logs(state_logs_dirs, labels, reward_tally_cutoff, imgs_dir, num_std=1):
    """
    Plot average and 2 standard deviation bands for each group of state logs.

    Args:
        state_logs_dirs (list of list of str): List of lists, where each sublist contains directories.
        labels (list of str): List of labels corresponding to each sublist in state_logs_dirs.
        reward_tally_cutoff (int): Cutoff time for the rewards.
        imgs_dir (str): Directory to save the image.
        num_std (int): Number of standard deviations away to plot the shaded region.
    """
    if len(state_logs_dirs) != len(labels):
        raise ValueError("The number of labels must match the number of sublists in state_logs_dirs.")
    
    tix_files_dir = os.path.join(imgs_dir, "reward_tix_files")
    os.makedirs(tix_files_dir, exist_ok=True)

    all_final_episode_cum_reward = []
    plt.figure(figsize=(10, 6))
    colors = plt.colormaps['tab10'].colors
    colors = np.array(colors).repeat(2, axis=0)
    for i, (dirs, label) in enumerate(zip(state_logs_dirs, labels)):
        all_data = []
        all_data_actual = []
        final_data_actual = []
        final_episode_cum_reward = []


        # Load data from all directories in the sublist
        for directory in dirs:
            # get parent directory of directory
            parent_dir = os.path.dirname(directory)
            eval_reward = pd.read_csv(os.path.join(parent_dir, "eval.csv"))
            eval_reward_actual = eval_reward["episode_reward"].to_numpy()
            train_reward = pd.read_csv(os.path.join(parent_dir, "train.csv"))
            train_reward_actual = train_reward["episode_reward"].to_numpy()
            full_reward_traj = np.load(os.path.join(directory, "rewards.npy"))
            full_reward_sum = np.sum(full_reward_traj, axis=1)[4:]
            avg_reward_sum = np.convolve(full_reward_sum, np.ones(5) / 5, mode='valid')
            idxs = []
            for eval_elem in eval_reward_actual:
                closest_idx = np.argmin(np.abs(avg_reward_sum - eval_elem))
                idxs.append(closest_idx)
            idxs = np.array(idxs)
            cutoff_full_reward = np.sum(full_reward_traj[4:, :reward_tally_cutoff], axis=1)
            avg_reward = np.convolve(cutoff_full_reward, np.ones(5) / 5, mode='valid')
            solved_rewards = avg_reward[idxs]
            
            partial_reward_traj = full_reward_traj[:, :reward_tally_cutoff]
            cum_partial_reward_traj = np.sum(partial_reward_traj, axis=1) # sum the rewards along each trajectory - shape: (num_episodes,)
            all_data.append(cum_partial_reward_traj)
            all_data_actual.append(eval_reward_actual)
            final_data_actual.append(solved_rewards)
            final_episode_cum_reward.append(cum_partial_reward_traj[-1])
        
        all_final_episode_cum_reward.append(final_episode_cum_reward)
        # Concatenate all data across directories
        all_data = np.array(all_data)  # shape: (num_directories, num_episodes)
        all_data_actual = np.array(all_data_actual)  # shape: (num_directories, num_episodes)
        all_final_data_actual = np.array(final_data_actual)  # shape: (num_directories, num_episodes)
        all_final_data_actual = all_final_data_actual / reward_tally_cutoff
        # Calculate mean and std dev
        mean = np.mean(all_final_data_actual, axis=0)
        std_dev = np.std(all_final_data_actual, axis=0)
        
        # Plot mean and 2*std_dev shaded region
        x = np.arange(mean.shape[0])  # Assuming sequential indices for x-axis
        line_style = "-"
        alpha = 0.3
        if i % 2 == 1:
            line_style = "--"
            alpha = 0.1
        plt.plot(x, mean, label=label, color=tuple(colors[i]), linestyle=line_style, alpha=alpha * 3)
        plt.fill_between(x, mean - num_std * std_dev, mean + num_std * std_dev, alpha=alpha, color=tuple(colors[i]))

        # Write data to file for tix plots
        curr_tix_filename = os.path.join(tix_files_dir, f"{label}.txt")
        x = np.arange(mean.shape[0])
        # write_data_to_file(x=x, mean=mean, std_dev=std_dev, filename=curr_tix_filename, num_std=num_std)
    
    plt.xlabel('Evaluation episodes')
    plt.ylabel('Avg. Reward')
    plt.ylim(0, 1)
    plt.xlim(0, 15)
    plt.legend(ncol=3, loc="upper center", fontsize=14)
    plt.title('State Logs with Mean and ±2 Standard Deviations')
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(os.path.join(imgs_dir, "partial_reward_graph.png"))
    plt.show()

    # Print the final episode cumulative rewards
    print("Final Episode Cumulative Rewards")
    for final_episode_rewards in all_final_episode_cum_reward: 
        print(final_episode_rewards)
    print(labels)
    print()
    return 


if __name__ == "__main__":
    imgs_dir =  "./images"
    reward_tally_cutoff = 650 # Cutoff time for the rewards
    num_std = 1
    os.makedirs(imgs_dir, exist_ok=True)

    # base_dir = "../exp/2024.11.28"
    # directories = [["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetTrue", "1501_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetTrue", ], 
    #                ["1010_safeCartpoleAvoid_swingup_sacCBF_safeswingup_14_12", "1219_safeCartpoleAvoid_swingup_sacCBF_safeswingup_15_12"], 
    #                ["1010_safeCartpoleRA_swingup_sac_test_exp_14", "1043_safeCartpoleRA_swingup_sac_test_exp_15"],
    #                ["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetFalse", "1516_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetFalse"]]

    base_dir = "../exp/2025.10.22"
    directories = [
                    ["noreset_seed13", "noreset", "noreset_seed33", "noreset_seed43", "noreset_seed53"], 
                    ["reset_seed13", "reset", "reset_seed33", "reset_seed43", "reset_seed53"], 
                    ["CBF_noreset_seed13", "CBF_noreset", "CBF_noreset_seed33", "CBF_noreset_seed43", "CBF_noreset_seed53"], 
                    ["CBF_reset_seed13", "CBF_reset", "CBF_reset_seed33", "CBF_reset_seed43", "CBF_reset_seed53"], 
                    ["RACBF_noreset_seed13", "RACBF_noreset_seed23", "RACBF_noreset_seed33", "RACBF_noreset_seed43", "RACBF_noreset_seed53"],
                    ["RACBF_reset_seed13", "RACBF_reset_seed23", "RACBF_reset_seed33", "RACBF_reset_seed43", "RACBF_reset_seed53"]
                    ]
    
    # base_dir = "/Users/nikhilushinde/Documents/Grad/research/arclab/RACBF_24/backup_results/all_state_logs_2024.11.29/"
    # directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
    #                ["0538_safeCartpoleAvoid_swingup_sacRACBF_swingup_43_5_resetTrue_spFalse", "2156_safeCartpoleAvoid_swingup_sacRACBF_swingup_53_5_resetTrue_spFalse", "1251_safeCartpoleAvoid_swingup_sacRACBF_swingup_33_5_resetTrue_spFalse", "1801_safeCartpoleAvoid_swingup_sacRACBF_swingup_13_5_resetTrue_spFalse", "2023_safeCartpoleAvoid_swingup_sacRACBF_swingup_23_5_resetTrue_spFalse"], #["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
    #                ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
    #                # NOTE: TODO: This last group is still incomplete! need to fix this later! 
    #                ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse", ]]

    post_dir = "state_logs"

    # Create full directory names 
    for sub_dir_num in range(len(directories)):
        for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
            directories[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), post_dir)
            assert(os.path.exists(directories[sub_dir_num][sub_sub_dir_num])), directories[sub_dir_num][sub_sub_dir_num]

    state_logs_dirs = directories 
    labels = [
        "SAC (w/o reset)",
        "SAC", 
        "SAC-CBF (w/o reset)",
        "SAC-CBF", 
        "Ours (w/o reset)",
        "Ours", 
            ]
            #   "sac-cbf", 
    #           "sac", 
    #           "sac-racbf-noreset"
    #           ]

    # Call the plotting function
    plot_state_logs(state_logs_dirs, labels, reward_tally_cutoff=reward_tally_cutoff, imgs_dir=imgs_dir, num_std=num_std)
    print("State logs plotted successfully.")