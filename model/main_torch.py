from ddpg_torch import Agent
from env import VortexENV
import numpy as np
from utils import plot
import math

# TODO: add a negative reward for going out of bounds or hitting the cylinder

env = VortexENV()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.001, env=env,
              batch_size=100, layer1_size=400, layer2_size=300, n_actions=1)

#agent.load_models()
np.random.seed(0)

score_history = []
mean_scores = []

# chkpt_dir = 'tmp/ddpg'
# if not os.path.exists(chkpt_dir):
#     os.makedirs(chkpt_dir)
#     print(f"Created directory {chkpt_dir} for saving checkpoints.")
    
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        try:
            temp = act * math.pi
            env.render(temp)
        except NameError:
            env.render()
        act = agent.choose_action(obs)[0]
        # print(act)
        new_state, reward, done, info = env.step(act)
        # print(reward)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    mean_scores.append(np.mean(score_history[-100:]))

    # if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    plot(score_history, mean_scores)