import cv2
import gym

env_name = 'LunarLander-v2'
env = gym.make(env_name)

print('{}'.format(env.observation_space))
print('{}'.format(env.action_space))

delay = 50

while True:
    observation = env.reset()
    total_reward = 0
    prev_action = 0
    while True:
        frame = env.render(mode='rgb_array')

        cv2.imshow(env_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(delay)

        # If ESC pressed, reset the environment
        if key == 27:
            break

        if key == 81:       # <-
            action = 1
        elif key == 82:     # ^
            action = 2
        elif key == 83:     # ->
            action = 3
        else:
            action = 0

        observation, reward, done, info = env.step(action)
        total_reward += reward

        prev_action = action

        if done:
            print(total_reward)
            break
