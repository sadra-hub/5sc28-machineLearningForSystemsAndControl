from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import time

from envs.UnbalancedDiskA2C import UnbalancedDisk


from stable_baselines3.common.env_util import make_vec_env

def make_my_env():
    return UnbalancedDisk()  # Your custom Gym environment class

vec_env = make_vec_env(make_my_env, n_envs=8)

model = A2C(
    "MlpPolicy",
    vec_env,
    verbose=0,
    device="cpu",
    ent_coef=0.01,  # 0.01
      # True

)


model.learn(total_timesteps=60_000, progress_bar=True) # 20_000 - 40_000
model.save("a2c")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c")
#model = A2C.load("./PTH/a2c_good")

env = UnbalancedDisk()
obs, info = env.reset()
try:
    for _ in range(1000):
        action, _states = model.predict(obs)
        #print(f'action={action}', f'obs={obs}') #debugging
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward}")
        env.render()
        time.sleep(1/10)
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()