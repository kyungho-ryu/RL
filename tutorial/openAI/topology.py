# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import gym

# .env : avoid training stopping at 200 iterations
env = gym.make("Taxi-v3").env

env.reset() # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# Setting a specific state
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

print(env.P[328])
# {action: [(probability, nextstate, reward, done)]}.