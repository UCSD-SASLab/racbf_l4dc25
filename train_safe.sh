Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
#python train_safe.py env=safePendulum_safeswingup seed=13
#python train_safe.py env=safePendulumDense_safeswingupdense seed=13
#python train_safe.py env=safePendulum_safeswingup seed=14
#python train_safe.py env=safePendulum_safeswingup seed=15
python train_safe.py env=safeCartpole_swingup seed=14
