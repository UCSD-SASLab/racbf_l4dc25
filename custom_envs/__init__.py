from dm_control import suite

# Custom environments
from custom_envs import safePendulum
from custom_envs import safePendulumDense
from custom_envs import safeCartpole
from custom_envs import safeCartpoleRA
from custom_envs import safeCartpoleAvoid

# Setup environments
suite._DOMAINS["safePendulum"] = safePendulum
suite._DOMAINS["safePendulumDense"] = safePendulumDense
suite._DOMAINS["safeCartpole"] = safeCartpole
suite._DOMAINS["safeCartpoleRA"] = safeCartpoleRA
suite._DOMAINS["safeCartpoleAvoid"] = safeCartpoleAvoid