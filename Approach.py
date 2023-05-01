from random import randint
from random import random

# Approach.py


# This program uses reinforcement learning to determine the optimal policy
# for Approach.
# Recall that approach works like this:
# Both players agree on a limit n.
# Player 1 rolls first. They go until they either exceed n or hold.
# Then player 2 rolls. They go until they either exceed n or beat player 1's score.
# The player who is closest to n without going over wins.
# Note:
# We can reduce this to the problem of player 1 choosing the best value at which to hold.
# This is called a policy; once we know the best number to hold at, we can act optimally.
def approach(n):
    q_table = [[random() / 100.0, random() / 100.0] for i in range(n + 1)]
    q_table[n][0] = 1
    q_table[n][1] = 0
    epsilon = 0.1
    alpha = 0.1
    gamma = 1

    for i in range(1000000):
        state = randint(0, n - 1)
        while True:
            best_action = q_table[state].index(max(q_table[state]))
            action = best_action if random() > epsilon else 1 - best_action

            reward, next_state = play_game(state, action, n)

            if next_state is not None:
                q_table[state][action] = q_table[state][action] + alpha * (
                            reward + gamma * max(q_table[next_state]) - q_table[state][action])
                state = next_state
            else:
                q_table[state][action] = q_table[state][action] + alpha * (reward - q_table[state][action])
                break

    for state in range(n + 1):
        optimal_action = "hold" if q_table[state].index(max(q_table[state])) == 0 else "roll"
        print(f"sum: {state}: hold {q_table[state][0]:.6f} roll {q_table[state][1]:.6f} [{optimal_action}]")


def play_game(state, action, n):
    if action == 0:  # hold
        opponent_state = 0
        while opponent_state < state and opponent_state <= n:
            roll = randint(1, 6)
            if opponent_state + roll <= n:
                opponent_state += roll
            else:
                break
        reward = 1 if opponent_state > n or opponent_state < state else 0
        next_state = None
    else:  # roll
        roll = randint(1, 6)
        if state + roll > n:
            reward = 0
            next_state = None
        else:
            reward = 0
            next_state = state + roll
            if next_state == n:
                reward = 1
                next_state = None

    return reward, next_state
