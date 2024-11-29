import numpy as np
import matplotlib.pyplot as plt
import os 

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
        data = np.load(file)  # Assumes each file contains a 2D numpy array
        all_data.append(data)
    return np.array(all_data)  # Convert to numpy array

def compute_average_violations(data):
    """
    Compute the average number of safety violations per episode.

    Args:
        data (np.ndarray): Array of safety violations (0 or 1). Shape: (num_episodes, length_of_episodes)

    Returns:
        float: Average number of safety violations per episode.
    """
    # return np.mean(np.sum(data, axis=1))  # Sum violations per episode, then average
    return np.mean(data)  # Sum violations per episode, then average

def plot_safety_violations(file_groups, labels, imgs_dir):
    """
    Plot a bar chart of average safety violations for each group.

    Args:
        file_groups (list of list of str): List of sublists, where each sublist contains file paths.
        labels (list of str): List of labels for each group.
    """
    if len(file_groups) != len(labels):
        raise ValueError("The number of labels must match the number of file groups.")
    
    averages = []
    
    for files in file_groups:
        data = load_data(files)
        avg_violations = compute_average_violations(data)
        averages.append(avg_violations)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, averages, color='skyblue', alpha=0.8)
    plt.xticks(x, labels)
    plt.xlabel('Groups')
    plt.ylabel('Average Safety Violations per Episode')
    plt.title('Average Safety Violations Across Groups')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(imgs_dir, "average_safety_violations.png"))


if __name__ == "__main__":
    imgs_dir =  "./images"
    reward_tally_cutoff = 1000 # Cutoff time for the rewards
    os.makedirs(imgs_dir, exist_ok=True)

    base_dir = "../exp/2024.11.28"
    directories = [["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetTrue", "1501_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetTrue", ], 
                   ["1010_safeCartpoleAvoid_swingup_sacCBF_safeswingup_14_12", "1219_safeCartpoleAvoid_swingup_sacCBF_safeswingup_15_12"], 
                   ["1010_safeCartpoleRA_swingup_sac_test_exp_14", "1043_safeCartpoleRA_swingup_sac_test_exp_15"],
                   ["1009_safeCartpoleRA_swingup_sacCBF_swingup_14_12_resetFalse", "1516_safeCartpoleRA_swingup_sacCBF_swingup_15_12_resetFalse"]]
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
    plot_safety_violations(file_groups, labels, imgs_dir=imgs_dir)
    print("State logs plotted successfully.")