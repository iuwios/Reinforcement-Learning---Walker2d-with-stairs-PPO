import gym
import gym_walker
import cv2

env=gym.make('Walker-v0')

while True:
	env.reset()
	env.step(env.action_space.sample())
	env.render()

