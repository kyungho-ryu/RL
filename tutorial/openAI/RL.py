import numpy as np
import gym, random
from tutorial.util import clear_output

env = gym.make("Taxi-v3").env

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

episodes = 10000
total_epochs, total_penalties = 0, 0

for i in range(1, episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

# Results after 100 episodes:
# Average timesteps per episode: 458.06
# Average penalties per episode: 24.65

# Results after 1000 episodes:
# Average timesteps per episode: 118.34
# Average penalties per episode: 4.952

# Results after 10000 episodes:
# Average timesteps per episode: 27.0296
# Average penalties per episode: 0.9336