from ddpg_torch import Agent
from env import VortexENV
import numpy as np
# from utils import plot
import math
import matplotlib.pyplot as plt

env = VortexENV()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.001, env=env,
              batch_size=100, layer1_size=400, layer2_size=300, n_actions=1)

#agent.load_models()
np.random.seed(0)

def setup_plots():
    plt.ion()  # Turn on interactive mode
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))  # Create two subplots side by side
    return fig, ax1

def update_plots(fig, ax1, episode, scores, avg_scores):
    ax1.clear()
    ax1.plot(scores, label='Score per Episode')
    ax1.plot(avg_scores, label='Average Score over last 100 Episodes')
    ax1.set_title('Score Progression')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    
    plt.pause(0.01)  # Pause briefly to update plots
    if episode % 100 == 0:  # Save the figure every 100 episodes
        plt.savefig(f'VortexEnv-ddpg-results-{episode}.png')

score_history = []

# fig, ax1 = setup_plots() # Setup plots

score_history = []
mean_scores = []
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
        new_state, reward, done, info = env.step(act)
        # print(reward)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    mean_scores.append(np.mean(score_history[-100:]))

    plt.plot(score_history)
    plt.plot(mean_scores)
    plt.show()

    #if i % 25 == 0:
    #    agent.save_models()
    # update_plots(fig, ax1, i, score_history, mean_scores)

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    # plot(score_history, mean_scores)
