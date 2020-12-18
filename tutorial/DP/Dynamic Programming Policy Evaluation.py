# https://github.com/dennybritz/reinforcement-learning

from IPython.core.debugger import set_trace
import numpy as np
import pprint
import sys

from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
env._render()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...

                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
# state가 지나감에 따라 next state value가 변해진다. 따라서 주변 state(next states)들의 값들이 작을 수록
# 현재 state값 또한 줄에 들게 된다.
# 그리고 episode가 지나가면서 주변 state가 전체적으로 줄어들어 현재 state가 지속적으로 감소한다.
# 마지막으로 delta로 줄어드는 정도를 체크하고 줄어드는 값이 theta보다 작아지면 training을 멈춘다.

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])

# Markov property: The environment's response at time t+1 depends only on the state and action representations at time t.
# The future is independent of the past given the present.

# Dynamic Programming (DP) methods assume that we have a perfect model of the environment's Markov Decision Process (MDP).
# That's usually not the case in practice, but it's important to study DP anyway.

# Policy Evaluation:
# Calculates the state-value function V(s) for a given policy.
# In DP this is done using a "full backup".
# At each state, we look ahead one step at each possible action and next state.
# We can only do this because we have a perfect model of the environment.
# Full backups are basically the Bellman equations turned into updates.