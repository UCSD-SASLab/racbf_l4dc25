Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
python train_safe_RA.py env=safeCartpoleRA_swingup seed=14
