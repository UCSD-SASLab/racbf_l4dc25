import numpy as np
import matplotlib.pyplot as plt
import os 

OFFSET = 4 # preseed trajectories to offset

def load_data(filenames):
    """
    Load data from a list of .npy files.

    Args:
        filenames (list of str): List of file paths.

    Returns:
        np.ndarray: Concatenated data from all files. Shape: (total_episodes, length_of_episodes)
    """
    all_data = []
    for file in filenames:
        data = np.load(file)[OFFSET:]  # Assumes each file contains a 2D numpy array
        all_data.append(data)
    return np.array(all_data)  # Convert to numpy array

def plot_avg_percent_unsafe_trajectories_pertraining(file_groups, labels, imgs_dir): 
    """
    A trajectory is unsafe if there is at least one safety violation
    Tally the number of unsafe trajectories/episodes during training and evaluation
    args: 
        file_groups (list of list of str): List of sublists, where each sublist contains file paths.
        labels (list of str): List of labels for each group.
        imgs_dir (str): Directory to save the image.
    """

    if len(file_groups) != len(labels):
        raise ValueError("The number of labels must match the number of file groups.")
    
    all_mean_unsafe_trajs = []
    all_std_unsafe_trajs = []
    
    for files in file_groups:
        data = load_data(files) # (num_files, num_episodes, episode_length)
        num_episodes = data.shape[1]

        unsafe_traj_level = np.sum(data, axis=2) # (num_files, num_episodes)
        num_unsafe_traj = np.sum(unsafe_traj_level > 0, axis=1) # (num_files,)
        avg_unsafe_traj = np.mean(num_unsafe_traj/num_episodes)
        std_unsafe_traj = np.std(num_unsafe_traj/num_episodes)

        all_mean_unsafe_trajs.append(avg_unsafe_traj)
        all_std_unsafe_trajs.append(std_unsafe_traj)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, all_mean_unsafe_trajs, yerr=all_std_unsafe_trajs, color='skyblue', alpha=0.8)
    plt.xticks(x, labels)
    plt.xlabel('Groups')
    plt.ylabel('Percent Unsafe Trajectories')
    plt.title('Percent Unsafe Trajectories during Training')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(imgs_dir, "average_percent_unsafe_trajs_per_episode.png"))
    # Printing the results
    print("Average Percentage of Unsafe Trajectories per Training")
    print("Mean Percents: ", all_mean_unsafe_trajs)
    print(labels)
    print("STD Percents: ", all_std_unsafe_trajs)
    print()


def plot_avg_unsafe_trajectories_pertraining(file_groups, labels, imgs_dir): 
    """
    A trajectory is unsafe if there is at least one safety violation
    Tally the number of unsafe trajectories/episodes during training and evaluation
    args: 
        file_groups (list of list of str): List of sublists, where each sublist contains file paths.
        labels (list of str): List of labels for each group.
        imgs_dir (str): Directory to save the image.
    """

    if len(file_groups) != len(labels):
        raise ValueError("The number of labels must match the number of file groups.")
    
    all_avg_unsafe_trajs = []
    
    for files in file_groups:
        data = load_data(files) # (num_files, num_episodes, episode_length)
        unsafe_traj_level = np.sum(data, axis=2) # (num_files, num_episodes)
        num_unsafe_traj = np.sum(unsafe_traj_level > 0, axis=1) # (num_files,)
        avg_unsafe_traj = np.mean(num_unsafe_traj)
        all_avg_unsafe_trajs.append(avg_unsafe_traj)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, all_avg_unsafe_trajs, color='skyblue', alpha=0.8)
    plt.xticks(x, labels)
    plt.xlabel('Groups')
    plt.ylabel('Average Unsafe Trajectories per Training')
    plt.title('Average Number of Unsafe Trajectories per Training')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(imgs_dir, "average_unasfe_trajs_per_training.png"))
    # Printing the results
    print("Average Unsafe Trajectories per Training")
    print(all_avg_unsafe_trajs)
    print(labels)
    print()

def avg_safety_violations_perepisode(file_groups, labels, imgs_dir): 
    """
    Compute the average number of safety violations per episode and plot them 
    args: 
        file_groups (list of list of str): List of sublists, where each sublist contains file paths.
        labels (list of str): List of labels for each group.
        imgs_dir (str): Directory to save the image.
    """
    if len(file_groups) != len(labels):
        raise ValueError("The number of labels must match the number of file groups.")
    
    all_avg_safety_violations = [] # for bar plot
    all_avg_safety_violations_lineplot = [] # for line plot 
    
    for files in file_groups:
        data = load_data(files) # (num_files, num_episodes, episode_length)
        num_safety_violations_per_episode = np.sum(data, axis=2) # (num_files, num_episodes)
        avg_safety_violations_per_episode = np.mean(num_safety_violations_per_episode, axis=0) # (num_episodes,)
        avg_safety_violations = np.mean(avg_safety_violations_per_episode)

        all_avg_safety_violations_lineplot.append(avg_safety_violations_per_episode)
        all_avg_safety_violations.append(avg_safety_violations)

    
    # Plotting Line Plot
    plt.figure(figsize=(10, 6))
    x = range(len(avg_safety_violations_per_episode))
    print("Episodes where safety violations occur:")
    for i, avg_safety_violations in enumerate(all_avg_safety_violations_lineplot): 
        plt.plot(x, avg_safety_violations, label=labels[i])
        print(f"{labels[i]}: {np.where(np.array(avg_safety_violations[:, 0])>0)}")
        print()
    print("Done with printing episodes where safety violations occur.\n\n")
    plt.xlabel('Episodes')
    plt.ylabel('Average Safety Violations per Episode')
    plt.title('Average Safety Violations per Episode Across Groups')
    plt.legend()
    # plt.grid(axis='y', linestyle='--', alpha=1.0)
    plt.savefig(os.path.join(imgs_dir, "average_safety_violations_lineplot.png"))

    # Plotting Bar Plot
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, all_avg_safety_violations, color='skyblue', alpha=0.8)
    plt.xticks(x, labels)
    plt.xlabel('Groups')
    plt.ylabel('Average Safety Violations per Episode')
    plt.title('Average Safety Violations Across Groups')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(imgs_dir, "average_safety_violations.png"))

    # Printing the results
    print("Average Safety Violations per Episode during Training")
    print(all_avg_safety_violations)
    print(labels)
    print()


if __name__ == "__main__":
    imgs_dir =  "./images"
    reward_tally_cutoff = 1000 # Cutoff time for the rewards
    os.makedirs(imgs_dir, exist_ok=True)

    # base_dir = "../exp/2024.11.29"
    # directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
    #                ["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
    #                ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
    #                # NOTE: TODO: This last group is still incomplete! need to fix this later! 
    #                ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse"]]

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

    post_dir = os.path.join("state_logs", "safety_violations.npy")

    # Create full directory names 
    for sub_dir_num in range(len(directories)):
        for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
            directories[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), post_dir)

    file_groups = directories 
    labels = ["sac-racbf", 
              "sac-cbf", 
              "sac", 
              "sac-racbf-noreset"]

    # Call the plotting function
    plot_avg_unsafe_trajectories_pertraining(file_groups, labels, imgs_dir=imgs_dir)
    avg_safety_violations_perepisode(file_groups, labels, imgs_dir=imgs_dir)
    plot_avg_percent_unsafe_trajectories_pertraining(file_groups, labels, imgs_dir=imgs_dir)
    print("State logs plotted successfully.")