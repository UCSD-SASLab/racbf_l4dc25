from dm_control import suite

# Custom environments
from custom_envs import safePendulum

# Setup environments
suite._DOMAINS["safePendulum"] = safePendulum