import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams.update({"font.family": "Monospace"})
EXPL = -1

def run_multiplicative_weights(n, m, w, q, friends, epsilon = 0.1, eta = 0.01, iterations = 10_000):
    log_weights = np.zeros((n, m)) 
    weights_over_time = np.zeros((n, iterations, m))
    
    for it in range(iterations):
        dist = np.zeros((n, m))
        if not (it < EXPL):
            for i in range(n):
                max_log = np.max(log_weights[i])
                stable_exp = np.exp(log_weights[i] - max_log)
                dist[i] = stable_exp / np.sum(stable_exp)
        else:
            dist = np.ones((n, m)) / m

        # Sample actions
        def get_actions(choices):
            if choices is None:
                choices = np.array([np.random.choice(m, p = dist[i]) for i in range(n)])
            counts = np.zeros(m, dtype = int)
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
            sucesssful_projects = np.zeros(m)

            # Calculate which projects were successful
            for i in range(n):
                if np.random.rand() < success_probs[i]:
                    sucesssful_projects[i] = 1
                    
            for i in range(n):
                chosen_project = choices[i]
                # Divide successful projects even amongst all players who chose it
                individual_payoffs[i] = w[chosen_project] * sucesssful_projects[chosen_project] / (counts[chosen_project])

            # Adding friends utility
            utilities = np.zeros(n)
            for i in range(n):
                total_payoff = individual_payoffs[i]
                friend_payoff = 0
                for f_i in friends[i]:
                    friend_payoff += individual_payoffs[f_i]

                total_payoff = total_payoff + epsilon * friend_payoff 
                utilities[i] = total_payoff   
            return choices, utilities
        
        choices, utilities = get_actions(None)
        for i in range(n):
            for j in range(m):
                if j == choices[i]:
                    log_weights[i, j] += eta * utilities[i]
                else:
                    new_choice = choices.copy()
                    new_choice[i] = j
                    _, new_utilities = get_actions(new_choice)
                    log_weights[i, j] += eta * (new_utilities[i])

        # for i in range(n):
        #     chosen_j = choices[i]
        #     log_weights[i, chosen_j] += eta * utilities[i]

        weights_over_time[:, it, :] = dist
        # plot_final_distribution(dist, m)
    
    # Return final distribution
    dist = np.zeros((n, m))
    for i in range(n):
        max_log = np.max(log_weights[i])
        stable_exp = np.exp(log_weights[i] - max_log)
        dist[i] = stable_exp / np.sum(stable_exp)
    return dist, weights_over_time

def plot_final_distribution(dist, m):
    """Assumes strategy converged to pure strategy and chooses the project with highest probability"""
    plt.figure(figsize=(12, 8))
    counts, bins, patches = plt.hist(np.argmax(dist, axis=1), bins=np.arange(0, m + 0.5, 1))

    plt.xlabel("Project Number")
    plt.ylabel("Number of Researchers on Project")
    plt.title("MW Algorithm: Final Distribution of Researchers on Projects")

    # Add numbers on top of the bars
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, int(count), ha = "center", va = "bottom")

    plt.show()

def plot_strategy_convergence(weights_over_time, n, m):
      rows = (n + 2) // 3 
      fig, axes = plt.subplots(rows, 3, figsize=(8, 2 * rows), sharex=True)
      axes = axes.flatten()  

      for i in range(n):
          for j in range(m):
            axes[i].plot(weights_over_time[i, :, j], label=f"Project {j + 1}")
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
    elif setup == 3:
        friend_mat = np.random.randint(0, 2, size=(N, N))
        np.fill_diagonal(friend_mat, 0)
        friend_mat = np.triu(friend_mat) + np.triu(friend_mat, 1).T
        friends = [list(np.where(friend_mat[i] == 1)[0]) for i in range(N)]
    return friends

def calculate_welfare(dist, w, q):
    ct = Counter(int(x) for x in np.argmax(dist, axis=1))
    welfare = 0
    for k, v in ct.items():
        welfare += (1 - (1 - q[k]) ** v) * w[k]
    return welfare
    
if __name__ == "__main__":
    # Contrived Example adapted from paper
    # np.random.seed(56)
    N = 16
    n = N
    m = N
    # w = np.array([1.0]  + [1/N] * (N - 1))
    # q = np.array([1.0] * N)
    
    # epsilon = [0, .25]
    tot_alt, tot_no_alt = 0, 0
    for i in range(10):
        w = np.random.rand(N)
        q = np.random.rand(N)
        friends = make_friends(n, setup = 2, num_friends = N - 1)
        dist_alt, _ = run_multiplicative_weights(n, m, w, q, friends, epsilon=.5, eta = .1, iterations = 1000)
        dist_no_alt, _ = run_multiplicative_weights(n, m, w, q, [[] for _ in range(N)], epsilon=0, eta = .1, iterations = 1000)
        tot_alt += calculate_welfare(dist_alt, w, q)
        tot_no_alt += calculate_welfare(dist_no_alt, w, q)
        print(f"Iter {i}: Cumulative with altruism: {tot_alt}, Cumulative without altruism: {tot_no_alt}")
    print(f"Average with altruism: {tot_alt/50}, Average without altruism: {tot_no_alt/50}")
    print(f"Average improvement with altruism: {tot_alt / tot_no_alt}")
    




    # print(friends)
    # Interesting Aside: Larger values of epsilon leads to faster convergence of strategy to pure strategy
    # plot_final_distribution(final_dist, m)
    # plot_strategy_convergence(strategy_over_time, n, m)
    # print("Final approximate distribution:")
    # df = pd.DataFrame(np.round(final_dist, 6), columns = [f"Project {i+1}" for i in range(m)], index = [f"Researcher {i+1}" for i in range(n)])
    # print(df)
