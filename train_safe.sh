# Xvfb :99 -screen 0 1400x900x24 &
# export DISPLAY=:99
Xvfb :105 -screen 0 800x600x24 &
export DISPLAY=:105
#python train_safe.py env=safePendulum_safeswingup seed=13
#python train_safe.py env=safePendulumDense_safeswingupdense seed=13
#python train_safe.py env=safePendulum_safeswingup seed=14
#python train_safe.py env=safePendulum_safeswingup seed=15
#python train_safe.py env=safeCartpole_swingup seed=14
python train_safe.py env=safeCartpoleAvoid_swingup seed=13 eval_frequency=12000 num_train_steps=200000
python train_safe.py env=safeCartpoleAvoid_swingup seed=23 eval_frequency=12000 num_train_steps=200000
python train_safe.py env=safeCartpoleAvoid_swingup seed=33 eval_frequency=12000 num_train_steps=200000
python train_safe.py env=safeCartpoleAvoid_swingup seed=43 eval_frequency=12000 num_train_steps=200000
python train_safe.py env=safeCartpoleAvoid_swingup seed=53 eval_frequency=12000 num_train_steps=200000
