import sys 
import os 

# Deepreach path: abstract away imports from deepreach 
deepreach_path = os.path.join(os.getcwd(), "../../")
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