# Xvfb :99 -screen 0 1400x900x24 &
Xvfb :121 -screen 0 800x600x24 &
export DISPLAY=:121

python train_safe_RA.py env=safeCartpoleRA_swingup seed=33 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=33 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true force_reset_env=false
python train_safe_RA.py env=safeCartpoleRA_swingup seed=43 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=43 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true force_reset_env=false
python train_safe_RA.py env=safeCartpoleRA_swingup seed=13 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=13 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true force_reset_env=false
python train_safe_RA.py env=safeCartpoleRA_swingup seed=53 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=53 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true force_reset_env=false
python train_safe_RA.py env=safeCartpoleRA_swingup seed=23 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=23 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true force_reset_env=false
