# Description: Test the avoid problem with time varying CBFs

# Xvfb :99 -screen 0 1400x900x24 &
# export DISPLAY=:99
Xvfb :105 -screen 0 800x600x24 &
export DISPLAY=:105

# Reach avoid agent for time varying
# safeCartpoleAvoid for only avoid problem 
python train_safe_RA.py env=safeCartpoleAvoid_swingup seed=53 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=false
python train_safe_RA.py env=safeCartpoleAvoid_swingup seed=43 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=false
python train_safe_RA.py env=safeCartpoleAvoid_swingup seed=33 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=false
python train_safe_RA.py env=safeCartpoleAvoid_swingup seed=23 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=false
python train_safe_RA.py env=safeCartpoleAvoid_swingup seed=13 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=false