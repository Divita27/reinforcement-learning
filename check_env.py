from stable_baselines3.common.env_checker import check_env
from env import VortexENV

# check env thorugh stable baselines
env = VortexENV()
check_env(env)

# check env again !
episodes = 50
for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:
		random_action = env.action_space.sample()
		print("action selected: ", random_action)
		obs, reward, done, info = env.step(random_action)
		print("reward: ", reward)