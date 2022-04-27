# Made possible by https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, x_label, y_label, title):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title(title + 'Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


