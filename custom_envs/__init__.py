from dm_control import suite

# Custom environments
from custom_envs import safePendulum
from custom_envs import safePendulumDense
from custom_envs import safeCartpole

# Setup environments
suite._DOMAINS["safePendulum"] = safePendulum
suite._DOMAINS["safePendulumDense"] = safePendulumDense
suite._DOMAINS["safeCartpole"] = safeCartpole