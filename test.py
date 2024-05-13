from ddpg_torch import Agent
from env_test import VortexENV
import numpy as np
from utils import plot
import math

env = VortexENV()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.001, env=env,
              batch_size=100, layer1_size=400, layer2_size=300, n_actions=1, test_mode=True)

agent.load_models(checkpoint_dir='tmp_old/ddpg')  
np.random.seed(0)

score_history = []
mean_scores = []

reach = 0

for i in range(30):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        # try:
        #     temp = act * math.pi
        #     env.render(temp)
        # except NameError:
        #     env.render()
        act = agent.choose_action(obs)[0]
        new_state, reward, done, info = env.step(act)
        if info:
            reach += 1
            print(f"Reached the goal {reach} times")
        score += reward
        obs = new_state

    score_history.append(score)
    mean_scores.append(np.mean(score_history[-100:]))

    print('episode ', i, 'score %.2f' % score, 'average score %.3f' % np.mean(score_history))
    
    plot(score_history, mean_scores, dirname = 'ddpg_test')
