import os 
import shutil
from copy import deepcopy

place_to_save = "./all_state_logs"
os.makedirs(place_to_save, exist_ok=True)

base_dir = "../exp/2024.11.29"
directories = [["0908_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetTrue", "0128_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetTrue", "0613_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetTrue", "1030_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetTrue", "1532_safeCartpoleRA_swingup_sacRACBF_swingup_53_5_resetTrue"], 
                ["0130_safeCartpoleAvoid_swingup_sacCBF_safeswingup_13_5", "0339_safeCartpoleAvoid_swingup_sacCBF_safeswingup_23_5", "0544_safeCartpoleAvoid_swingup_sacCBF_safeswingup_33_5", "0747_safeCartpoleAvoid_swingup_sacCBF_safeswingup_43_5", "0952_safeCartpoleAvoid_swingup_sacCBF_safeswingup_53_5"], 
                ["0135_safeCartpoleRA_swingup_sac_test_exp_13", "0207_safeCartpoleRA_swingup_sac_test_exp_23", "0240_safeCartpoleRA_swingup_sac_test_exp_33", "0313_safeCartpoleRA_swingup_sac_test_exp_43", "0346_safeCartpoleRA_swingup_sac_test_exp_53"], 
                # NOTE: TODO: This last group is still incomplete! need to fix this later! 
                ["0133_safeCartpoleRA_swingup_sacRACBF_swingup_13_5_resetFalse", "0638_safeCartpoleRA_swingup_sacRACBF_swingup_23_5_resetFalse", "1142_safeCartpoleRA_swingup_sacRACBF_swingup_33_5_resetFalse", "1539_safeCartpoleRA_swingup_sacRACBF_swingup_43_5_resetFalse"]]
full_dir_paths = deepcopy(directories)
post_dir = "state_logs"

# Create full directory names 
for sub_dir_num in range(len(directories)):
    for sub_sub_dir_num in range(len(directories[sub_dir_num])): 
        full_dir_paths[sub_dir_num][sub_sub_dir_num] = os.path.join(os.path.join(base_dir, directories[sub_dir_num][sub_sub_dir_num]), post_dir)
        assert(os.path.exists(directories[sub_dir_num][sub_sub_dir_num])), directories[sub_dir_num][sub_sub_dir_num]


for sub_dir_num in range(len(full_dir_paths)):
    for sub_sub_dir_num in range(len(full_dir_paths[sub_dir_num])): 
        local_dir = os.path.join(place_to_save, os.path.join(directories[sub_dir_num][sub_sub_dir_num], post_dir))
        shutil.copytree(full_dir_paths[sub_dir_num][sub_sub_dir_num], local_dir)
        print(f"Copying {full_dir_paths[sub_dir_num][sub_sub_dir_num]} to {local_dir}")

print("Done copying all state logs!")