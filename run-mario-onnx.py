import torch
import time
import random
import sys
import gymnasium as gym
import numpy as np
import cv2
import onnxruntime as ort
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from collections import deque

GAME_NAME = 'SuperMarioBros-1-1-v1'
FRAME_STACK = 4
MAX_EPISODE_STEPS = 10000

np.set_printoptions(suppress=True)

def preprocess_state(state):
    # 裁掉左侧16列
    state = state[:, 16:, :]
    # 转为灰度图并缩放到80x80
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class FrameStacker:
    def __init__(self, k: int = FRAME_STACK):
        self.k = k
        self.frames = deque(maxlen=k)
        self._stacked = np.zeros((k, 80, 80), dtype=np.float32)

    def reset(self, frame):
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get_stacked()

    def step(self, frame):
        self.frames.append(frame)
        return self._get_stacked()

    def _get_stacked(self):
        for i, f in enumerate(self.frames):
            self._stacked[i] = f
        return self._stacked


def run_onnx_model(model_path, num_episodes=10):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    env = JoypadSpace(gym_super_mario_bros.make(GAME_NAME, max_episode_steps=MAX_EPISODE_STEPS), SIMPLE_MOVEMENT)
    frame_stacker = FrameStacker(FRAME_STACK)

    rewards = []
    for i in range(num_episodes):
        state, _ = env.reset()
        state = frame_stacker.reset(preprocess_state(state))
        total_reward = 0
        done = False
        steps = 0
        action = 0
        while not done:
            infer = bool(steps % 1 == 0)
            if infer:
                input_data = state.astype(np.float32).reshape(1, FRAME_STACK, 80, 80)
                outputs = session.run(None, {input_name: input_data})
                action = torch.distributions.Categorical(logits=torch.tensor(outputs[-1])).sample().item()
                # action = np.argmax(outputs[-1])
                # action = random.randint(0, 6)
            else:
                action = 0
            # print(f"step={steps}, action={action}")
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(0.01)
            state = frame_stacker.step(preprocess_state(next_state))
            steps += 1
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"test.episode={i}, steps={steps}, reward={total_reward}")
    print(f"reward.mean={sum(rewards) / len(rewards):.3f}")
    env.close()


if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "mario-a2c-1.onnx"
    try:
        run_onnx_model(model_file)
    except Exception as e:
        print(f"error: {e}")
