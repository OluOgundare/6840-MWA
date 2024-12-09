import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Monospace'})

def run_multiplicative_weights(n, m, w, q, friends, epsilon = 0.1, eta = 0.01, iterations = 10_000):
    log_weights = np.zeros((n, m)) 
    weights_over_time = np.zeros((n, iterations, m))
    
    for it in range(iterations):
        dist = np.zeros((n, m))
        for i in range(n):
            max_log = np.max(log_weights[i])
            stable_exp = np.exp(log_weights[i] - max_log)
            dist[i] = stable_exp / np.sum(stable_exp)

        # Sample actions
        choices = np.array([np.random.choice(m, p = dist[i]) for i in range(n)])
        counts = np.zeros(m, dtype=int)
        for c in choices:
            counts[c] += 1

        success_probs = np.zeros(m)
        for j in range(m):
            if counts[j] > 0:
                f_j = 1 - q[j]
                success_probs[j] = 1 - (f_j ** counts[j])
            else:
                success_probs[j] = 0

        individual_payoffs = np.zeros(n)
        for i in range(n):
            chosen_project = choices[i]
            if success_probs[chosen_project] > 1e-12:
                  if np.random.rand() < success_probs[chosen_project]:
                        individual_payoffs[i] = w[chosen_project] / counts[chosen_project]
                  else:
                        individual_payoffs[i] = 0.0

        # Adding friends utility
        utilities = np.zeros(n)
        for i in range(n):
            total_payoff = individual_payoffs[i]
            friend_payoff = 0
            for f_i in friends[i]:
                friend_payoff += individual_payoffs[f_i]

            total_payoff = total_payoff + epsilon * friend_payoff 
            utilities[i] = total_payoff

        utilities = np.clip(utilities, -10, 10)

        for i in range(n):
            chosen_j = choices[i]
            log_weights[i, chosen_j] += eta * utilities[i]

        weights_over_time[:, it, :] = dist
    
    # Return final distribution
    dist = np.zeros((n, m))
    for i in range(n):
        max_log = np.max(log_weights[i])
        stable_exp = np.exp(log_weights[i] - max_log)
        dist[i] = stable_exp / np.sum(stable_exp)
    return dist, weights_over_time

def plot_final_distribution(dist, m):
    """Assumes strategy convered to pure strategy and chooses the project with highest probability"""
    plt.figure(figsize = (12, 8))
    plt.hist(np.argmax(dist, axis = 1), bins = np.arange(0, m + 0.5, 1))

    plt.xlabel("Project Number")
    plt.ylabel("Number of Researchers on Prokect")
    plt.title("MW Algorithm: Final Distribution of Researchers on Projects")
    plt.show()

def plot_strategy_convergence(weights_over_time, n, m):
      rows = (n + 2) // 3 
      fig, axes = plt.subplots(rows, 3, figsize=(8, 2 * rows), sharex=True)
      axes = axes.flatten()  

      for i in range(n):
          for j in range(m):
            axes[i].plot(weights_over_time[i, :, j], label=f"Project {j}")
            axes[i].set_title(f"Player {i + 1} Strategy Convergence")
            axes[i].set_ylabel("Probability")
            axes[i].legend()

      for ax in axes[n:]: 
            ax.set_visible(False)

      plt.xlabel("Iteration")
      plt.tight_layout()
      plt.show()

def make_friends(n, setup, num_friends = 3):
    """setup = 0: no friends, 1: best friends, 2: everyone has same number of friends"""
    if setup == 0:
      return [[] for _ in range(n)]
    elif setup == 1:
      return [[i + 1] if i % 2 == 0 else [i - 1] for i in range(n)]
    elif setup == 2:
      friends = []
      for i in range(n):
          friends.append([(i + j) % n for j in range(1, num_friends + 1)])
      return friends
    
if __name__ == "__main__":
    n = 6
    m = 5
    w = np.array([10, 20, 5, 15, 25])
    q = np.array([0.9, 0.2, 0.05, 0.15, 0.3])
    epsilon = 0.1 # Altruism Factor: [0,1)
    friends = make_friends(n, setup = 1)

    # Interesting Aside: Larger values of epsilon leads to faster convergence to pure strategy
    final_dist, strategy_over_time = run_multiplicative_weights(n, m, w, q, friends, epsilon, eta=0.01, iterations=100)
    plot_final_distribution(final_dist, m)
    plot_strategy_convergence(strategy_over_time, n, m)
    print("Final approximate distribution:")
    df = pd.DataFrame(np.round(final_dist, 6), columns = [f"Project {i+1}" for i in range(m)], index = [f"Researcher {i+1}" for i in range(n)])
    print(df)
