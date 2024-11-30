# Xvfb :99 -screen 0 1400x900x24 &
Xvfb :113 -screen 0 800x600x24 &
export DISPLAY=:113
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=14
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=13 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=23 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=33 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=43 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=53 eval_frequency=12000 num_train_steps=200000

# python train_safe_RA.py env=safeCartpoleRA_swingup seed=16 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=17 eval_frequency=12000 num_train_steps=200000
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=18 eval_frequency=12000 num_train_steps=200000

python train_safe_RA.py env=safeCartpoleRA_swingup seed=13 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=23 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=33 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=43 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true
python train_safe_RA.py env=safeCartpoleRA_swingup seed=53 eval_frequency=12000 num_train_steps=200000 agent.params.cbf_alpha_value=5 safe_pre_seed=true