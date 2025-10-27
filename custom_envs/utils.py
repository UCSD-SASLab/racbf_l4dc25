import sys 
import os 

# Deepreach path: abstract away imports from deepreach 
# deepreach_path = os.path.join(os.getcwd(), "../../")
deepreach_path = os.path.join(os.path.dirname(__file__), "../../../")
try: 
    assert("deepreach" in os.listdir(deepreach_path))
except AssertionError as e: 
    print(os.listdir(deepreach_path))
    import pdb; pdb.set_trace()
    print("Need to modify deepreach path in custom_envs/utils.py")
    print("Deepreach Path assertion failed: ", e)
    raise 
sys.path.append(deepreach_path)

# Deepreach imports 
from deepreach.dynamics.dynamics import InvertedPendulum as InvertedPendulumDeepreach
from deepreach.dynamics.dynamics_hjr import InvertedPendulum as InvertedPendulumHJR

from deepreach.dynamics.dynamics import CartPole as CartPoleDeepreach
from deepreach.dynamics.dynamics_hjr import CartPole as CartPoleHJR

from deepreach.dynamics.dynamics import BaseCartPole as BaseCartPoleDeepreach
from deepreach.dynamics.dynamics_hjr import BaseCartPole as BaseCartPoleHJR