import matplotlib.pyplot as plt 
import numpy as np

# def plotLearning(scores, filename, x=None, window=5):   
#     N = len(scores)
#     running_avg = np.empty(N)
        
#     for t in range(N):
# 	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
     
#     if x is None:
#         x = [i for i in range(N)]

#     plt.ylabel('Score')       
#     plt.xlabel('Game')                     
#     plt.plot(x, running_avg)
#     plt.savefig(filename)

import matplotlib.pyplot as plt

def plotLearning(score, avg_scores, actions, filename, window=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots side by side

    ax1.plot(score, label='Score per Episode')
    ax1.plot(avg_scores, label='Average Score over last {} Episodes'.format(window))
    ax1.set_title('Score Progression')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()

    ax2.plot(actions, label='Average Action Value')
    ax2.set_title('Action Value Progression')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Action Value')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
