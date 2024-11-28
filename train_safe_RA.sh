# Xvfb :99 -screen 0 1400x900x24 &
Xvfb :99 -screen 0 800x600x24 &
export DISPLAY=:99
# python train_safe_RA.py env=safeCartpoleRA_swingup seed=14
python train_safe_RA.py env=safeCartpoleRA_swingup seed=14 eval_frequency=12000 num_train_steps=200000
python train_safe_RA.py env=safeCartpoleRA_swingup seed=15 eval_frequency=12000 num_train_steps=200000
python train_safe_RA.py env=safeCartpoleRA_swingup seed=16 eval_frequency=12000 num_train_steps=200000
python train_safe_RA.py env=safeCartpoleRA_swingup seed=17 eval_frequency=12000 num_train_steps=200000
python train_safe_RA.py env=safeCartpoleRA_swingup seed=18 eval_frequency=12000 num_train_steps=200000
python train_safe_RA.py env=safeCartpoleRA_swingup seed=19 eval_frequency=12000 num_train_steps=200000
