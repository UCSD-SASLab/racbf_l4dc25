Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
#python train.py env=cheetah_run  
#python train.py env=pendulum_swingup
#python train.py env=cartpole_swingup
#python train.py env=pendulum_swingup seed=1300 num_train_steps=100000000 eval_frequency=5000000 log_frequency=5000000 &>> truncated_logfile_1300_longer.txt
#python train.py env=pendulum_swingup seed=1300 num_train_steps=1e8 eval_frequency=100000 log_frequency=100000 
#python train.py env=pendulum_swingup seed=21 &>> normal_swingup_logforrange_21.txt
#python train.py env=cartpole_swingup seed=11
#python train.py env=cartpole_swingup_sparse seed=11
#python train.py env=safeCartpole_swingup seed=13
python train.py env=safeCartpoleRA_swingup seed=14 eval_frequency=12000
