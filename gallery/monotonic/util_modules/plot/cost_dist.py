# costs 분포
import matplotlib.pyplot as plt
import numpy as np



def plot_cost_distribution(costs, bins=50):
    
    # Convert the negative log costs back to positive costs for better interpretability
    # costs = np.exp(-costs)
    plt.hist(costs, bins=bins)
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.title("Distribution of Costs")
    plt.show()


