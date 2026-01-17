import time
import gymnasium as gym
import numpy as np
from rich import print

# 0: 向上移动
# 1: 向右移动
# 2: 向下移动
# 3: 向左移动

gamma = 1
lr = 0.1  # learning rate


def run_game(episodes: int = 1000, max_steps: int = 1000):
    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    print(f"env.observation_space={env.observation_space}")
    print(f"env.action_space={env.action_space}")
    table = np.zeros((env.observation_space.n, env.action_space.n))

    for ep in range(1, episodes + 1):
        observation, info = env.reset(seed=ep)
        total_reward = 0.0

        for t in range(max_steps):
            val = np.random.uniform(0, 1)
            if val < 0.75:
                action = env.action_space.sample()
            else:
                maxval = np.max(table[observation])
                action = np.random.choice(np.where(table[observation] == maxval)[0])
            observation_next, reward, terminated, truncated, info = env.step(action)
            # print(f"observation={observation}")
            total_reward += reward

            # add code here
            td_target = reward + gamma * np.max(table[observation_next])
            table[observation, action] += lr * (td_target - table[observation, action])
            observation = observation_next

            # check exit
            if terminated or truncated:
                break
        print(f"train={ep}, total_reward={total_reward:.1f} steps={t+1}")

    print(f"table=\n{table}")

    print("-" * 32)
    dir_table = ["↑", "→", "↓", "←"]
    for i in range(table.shape[0]):
        maxval = np.max(table[i])
        if len(np.where(table[i] == maxval)[0]) == 4:
            print("*", end=" ")
        else:
            print(f"{dir_table[np.argmax(table[i])]}", end=" ")
        if (i > 0) and (((i + 1) % 12) == 0):
            print()
    env.close()

    # test
    env = gym.make("CliffWalking-v1", render_mode="human")
    while True:
        observation, info = env.reset(seed=ep)
        total_reward = 0.0
        for t in range(max_steps):
            maxval = np.max(table[observation])
            action = np.random.choice(np.where(table[observation] == maxval)[0])
            observation_next, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            observation = observation_next
            if terminated or truncated:
                break
        print(f"test.total_reward={total_reward:.1f} steps={t+1}")

    env.close()


if __name__ == "__main__":
    print("Running a simple random CliffWalking-v1 demo using Gymnasium\n")
    start = time.time()
    run_game(episodes=1000)
    print(f"\nDone — runtime: {time.time() - start:.2f}s")
