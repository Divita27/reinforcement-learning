from stable_baselines3 import A2C
from env import VortexENV

# Parallel environments
env = VortexENV()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50)
model.save("a2c_khalasi")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")