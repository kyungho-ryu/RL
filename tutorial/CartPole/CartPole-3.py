import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
# Discrete = space allows a fixed range of non-negative numbers, (0,1)

print(env.observation_space)
#> Box(4,)
# Box = Dimensional Box

print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
# Box Boundary


from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
print(x)
assert space.contains(x)
assert space.n == 8
