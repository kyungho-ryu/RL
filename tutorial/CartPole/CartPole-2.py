import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    # env.reset() - returns an initial observation
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        # env.step() - returns observation, reward, done, and info
        # done = done being True indicates the episode has terminated.
        #        (For example, perhaps the pole tipped too far, or you lost your last life.)
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("TEST, ", observation, reward)
            break
env.close()