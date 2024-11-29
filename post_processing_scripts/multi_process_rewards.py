import matplotlib.pyplot as plt
import numpy as np
import os

def plot_state_logs(state_logs_dirs, labels, reward_tally_cutoff, imgs_dir):
    """
    Plot average and 2 standard deviation bands for each group of state logs.

    Args:
        state_logs_dirs (list of list of str): List of lists, where each sublist contains directories.
        labels (list of str): List of labels corresponding to each sublist in state_logs_dirs.
    """
    if len(state_logs_dirs) != len(labels):
        raise ValueError("The number of labels must match the number of sublists in state_logs_dirs.")
    
    plt.figure(figsize=(10, 6))
    
    for i, (dirs, label) in enumerate(zip(state_logs_dirs, labels)):
        all_data = []
        
        # Load data from all directories in the sublist
        for directory in dirs:
            full_reward_traj = np.load(os.path.join(directory, "rewards.npy"))
            partial_reward_traj = full_reward_traj[:, :reward_tally_cutoff]
            cum_partial_reward_traj = np.sum(partial_reward_traj, axis=1) # sum the rewards along each trajectory - shape: (num_episodes,)
            all_data.append(cum_partial_reward_traj)
        
        # Concatenate all data across directories
        all_data = np.array(all_data)  # shape: (num_directories, num_episodes)
        
        # Calculate mean and std dev
        mean = np.mean(all_data, axis=0)
        std_dev = np.std(all_data, axis=0)
        
        # Plot mean and 2*std_dev shaded region
        x = np.arange(mean.shape[0])  # Assuming sequential indices for x-axis
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - 2 * std_dev, mean + 2 * std_dev, alpha=0.3)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.title('State Logs with Mean and ±2 Standard Deviations')
    plt.grid(True)
    plt.savefig(os.path.join(imgs_dir, "partial_reward_graph.png"))
    return 


if __name__ == "__main__":
    imgs_dir =  "./images"
    reward_tally_cutoff = 1000 # Cutoff time for the rewards
    os.makedirs(imgs_dir, exist_ok=True)

    base_dir = "../exp/2024.11.28"
    directories = [["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetTrue", "1501_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetTrue", ], 
                   ["1010_safeCartpoleAvoid_swingup_sacCBF_safeswingup_14_12", "1219_safeCartpoleAvoid_swingup_sacCBF_safeswingup_15_12"], 
                   ["1010_safeCartpoleRA_swingup_sac_test_exp_14", "1043_safeCartpoleRA_swingup_sac_test_exp_15"],
                   ["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetFalse", "1516_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetFalse"]]
    post_dir = "state_logs"

    # Create full directory names 
    for sub_dir_num in range(len(directories)):
        for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
            directories[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), post_dir)

    state_logs_dirs = directories 
    labels = ["sac-racbf", 
              "sac-cbf", 
              "sac", 
              "sac-racbf-noreset"]

    # Call the plotting function
    plot_state_logs(state_logs_dirs, labels, reward_tally_cutoff=reward_tally_cutoff, imgs_dir=imgs_dir)
    print("State logs plotted successfully.")