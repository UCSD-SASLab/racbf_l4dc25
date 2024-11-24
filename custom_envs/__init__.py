from dm_control import suite

# Custom environments
from custom_envs import safePendulum
from custom_envs import safePendulumDense

# Setup environments
suite._DOMAINS["safePendulum"] = safePendulum
suite._DOMAINS["safePendulumDense"] = safePendulumDense