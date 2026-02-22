import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from typing import cast
from collections import deque
from rich import print

import onnx
from onnx import numpy_helper

GAME_NAME = 'SuperMarioBros-1-1-v1'

# Hyperparameters
FRAME_STACK = 4
GAMMA = 0.99  # 提高 gamma 以更好地传播长期奖励
GAE_LAMBDA = 0.95  # GAE lambda 参数，平衡 bias-variance
LEARNING_RATE = 1e-3
TRAIN_EPISODES = 10000
GRADIENT_CLIP = 0.5
VALUE_LOSS_COEF = 1.0
ADV_NORM_EPS = 1e-8
ENTROPY_BETA_START = 0.1
ENTROPY_BETA_END = 0.0005
ENTROPY_BETA_DECAY = 0.999
MAX_EPISODE_STEPS = 2048

# 二次训练
LEARNING_RATE = 1e-5
ENTROPY_BETA_START = 5e-4
ENTROPY_BETA_END = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# last_second = 0

class SimpleTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"timer.{self.name}.elapsed={elapsed_time:.6f}s")

def preprocess_state(state):
    # global last_second
    # 裁掉左侧16列
    state = state[:, 16:, :]
    # 转为灰度图并缩放到80x80
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
    # curr_second = int(time.time())
    # if curr_second != last_second:
    #     last_second = curr_second
    #     cv2.imshow("State", resized)
    #     cv2.waitKey(1)
    return resized.astype(np.float32) / 255.0


def export_onnx(policy_net, filename="mario-a2c-1.onnx"):
    was_training = policy_net.training
    policy_net.eval()
    dummy_input = torch.randn(1, FRAME_STACK, 80, 80).to(device)
    try:
        torch.onnx.export(
            policy_net,
            (dummy_input,),
            filename,
            input_names=["input"],
            output_names=["value", "action_probs"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "value": {0: "batch_size"},
                "action_probs": {0: "batch_size"},
            },
        )
    finally:
        if was_training:
            policy_net.train()
    print(f"\nModel exported to {filename}")


class FrameStacker:
    """堆叠连续帧以捕获时序信息（运动方向、速度等）"""

    def __init__(self, k: int = FRAME_STACK):
        self.k = k
        self.frames = deque(maxlen=k)
        self._stacked = np.zeros((k, 80, 80), dtype=np.float32)

    def reset(self, frame):
        """重置时用同一帧填充"""
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get_stacked()

    def step(self, frame):
        """添加新帧，返回堆叠结果"""
        self.frames.append(frame)
        return self._get_stacked()

    def _get_stacked(self):
        for i, f in enumerate(self.frames):
            self._stacked[i] = f
        return self._stacked


class EntropyScheduler:
    def __init__(self, start, end, decay):
        self.beta = start
        self.end = end
        self.decay = decay

    def step(self):
        self.beta = max(self.end, self.beta * self.decay)
        return self.beta

    def get_beta(self):
        return self.beta


def compute_gae(rewards, values, next_values, terminateds, gamma, lam):
    """Compute Generalized Advantage Estimation (GAE).

    A_t = sum_{l=0}^{inf} (gamma * lam)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
    """
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - terminateds[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - terminateds[t]) * gae
        advantages.insert(0, gae)
    return advantages


class ActorCriticNet(torch.nn.Module):
    def __init__(self, action_dim):
        super(ActorCriticNet, self).__init__()
        self.action_dim = action_dim
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.feature_net = torch.nn.Sequential(
            self.conv_net,
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64 * 6 * 6, 512),
            torch.nn.ReLU(),
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, int(action_dim)),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_net(features)
        logits = self.policy_net(features)
        return value, logits

    def action_probs(self, x):
        features = self.feature_net(x)
        return torch.nn.functional.softmax(self.policy_net(features), dim=-1)

    def init_from_onnx(self, onnx_path: str):
        # 检查 ONNX 模型是否存在
        if not onnx_path or not os.path.isfile(onnx_path):
            print(f"ONNX model file not found: {onnx_path}")
            return self

        onnx_model = onnx.load(onnx_path)
        state_dict = self.state_dict()

        loaded_keys = []
        skipped_missing = []
        skipped_shape = []

        for init in onnx_model.graph.initializer:
            key = init.name
            if key not in state_dict:
                skipped_missing.append(key)
                continue

            tensor = torch.from_numpy(np.array(numpy_helper.to_array(init), copy=True)).to(dtype=state_dict[key].dtype)
            if state_dict[key].shape != tensor.shape:
                skipped_shape.append((key, tuple(tensor.shape), tuple(state_dict[key].shape)))
                continue

            state_dict[key].copy_(tensor)
            loaded_keys.append(key)

        print(f"Loaded {len(loaded_keys)} tensors from ONNX: {onnx_path}")
        if skipped_missing:
            print(f"Skipped missing keys ({len(skipped_missing)}): {skipped_missing[:5]}")
        if skipped_shape:
            print(f"Skipped shape-mismatch keys ({len(skipped_shape)}): {skipped_shape[:5]}")

        return self


class Agent:
    def __init__(self, action_dim):
        self.model = ActorCriticNet(action_dim).init_from_onnx("./12000.onnx").to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
        self.entropy_scheduler = EntropyScheduler(ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY)

        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.next_states = []
        self.terminateds = []

    def select_action(self, state):
        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
        value, logits = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()

    def store_transition(self, log_prob, value, reward, entropy, next_state, terminated):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminateds.append(float(terminated))

    def update_batch(self):
        if len(self.rewards) == 0:
            return

        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)
        entropies = torch.cat(self.entropies)
        rewards = self.rewards
        terminateds = self.terminateds

        with torch.no_grad():
            last_state = np.stack([self.next_states[-1]])
            last_state_t = torch.tensor(last_state, dtype=torch.float, device=device)
            last_value, _ = self.model(last_state_t)
            last_value = last_value.squeeze(-1).item()

        values_list = [v.item() for v in values]
        next_values = values_list[1:] + [last_value]

        advantages = compute_gae(
            rewards=rewards,
            values=values_list,
            next_values=next_values,
            terminateds=terminateds,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )
        advantages = torch.tensor(advantages, dtype=torch.float, device=device)
        adv_norm = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + ADV_NORM_EPS)

        targets = advantages + values

        entropy_beta = self.entropy_scheduler.get_beta()
        critic_loss = torch.nn.functional.mse_loss(values, targets)
        actor_loss = -(log_probs * adv_norm).mean()
        entropy_loss = -entropies.mean() * entropy_beta
        loss = actor_loss + VALUE_LOSS_COEF * critic_loss + entropy_loss
        # print(f"critic_loss={critic_loss.item():.6f}", end='\t')
        # print(f"actor_loss={actor_loss.item():.6f}", end='\t')
        # print(f"entropy_loss={entropy_loss.item():.6f}", end='\t')
        # print(f"total_loss={loss.item():.6f}", end='\t')
        # print()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

        self.clear_memory()

        return critic_loss.item(), actor_loss.item(), entropy_loss.item(), loss.item()

    def clear_memory(self):
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.next_states = []
        self.terminateds = []

    def step_scheduler(self):
        current_lr = self.scheduler.get_last_lr()[0]
        if current_lr > self.scheduler.eta_min:
            self.scheduler.step()
        return current_lr


def train(episodes=TRAIN_EPISODES):
    env = JoypadSpace(gym_super_mario_bros.make(GAME_NAME, max_episode_steps=MAX_EPISODE_STEPS), SIMPLE_MOVEMENT)
    observation_space = cast(Box, env.observation_space)
    action_space = cast(Discrete[np.int64], env.action_space)
    assert observation_space.shape is not None
    state_dim = observation_space.shape
    action_dim = int(action_space.n)
    print(f"state_dim={state_dim}, action_dim={action_dim}")
    agent = Agent(action_dim)
    frame_stacker = FrameStacker(FRAME_STACK)
    hist_rewards = deque(maxlen=10)

    for episode in range(episodes):
        state, _ = env.reset()
        state = frame_stacker.reset(preprocess_state(state))
        total_reward = 0.0
        total_entropy = 0.0
        steps = 0
        done = False
        action = None

        while not done:
            infer = bool(steps % 4 == 0)
            if infer:
                action, log_prob, value, entropy = agent.select_action(state)
            else:
                action = 0 # 选择 NOOP 动作以保持状态更新，但不影响游戏
            next_state_raw, reward_raw, terminated, truncated, info = env.step(action)
            if terminated:
                reward_raw -= (MAX_EPISODE_STEPS - steps - 1) * 0.1
            if infer:
                state = frame_stacker.step(preprocess_state(next_state_raw))
            reward = float(reward_raw) / 100.0
            if infer:
                agent.store_transition(
                    log_prob,
                    value,
                    reward,
                    entropy,
                    state,
                    bool(terminated),
                )
            total_reward += reward
            total_entropy += float(entropy)
            steps += 1
            done = terminated or truncated

        critic_loss, actor_loss, entropy_loss, total_loss = (0.0, 0.0, 0.0, 0.0)
        if len(agent.rewards) > 0:
            critic_loss, actor_loss, entropy_loss, total_loss = agent.update_batch()

        hist_rewards.append(total_reward)
        mean_reward = sum(hist_rewards) / len(hist_rewards)
        mean_entropy = total_entropy / steps
        entropy_beta = agent.entropy_scheduler.get_beta()
        lr = agent.step_scheduler()
        now = time.time()
        ts = time.strftime('%H:%M:%S', time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        print(ts, end='\t')
        print(f"episode={episode}", end='\t')
        print(f"steps={steps}", end='\t')
        print(f"curr_reward={total_reward:.3f}", end='\t')
        print(f"mean_reward={mean_reward:.3f}", end='\t')
        print(f"mean_entropy={mean_entropy:.3f}", end='\t')
        print(f"critic_loss={critic_loss:.6f}", end='\t')
        print(f"actor_loss={actor_loss:.6f}", end='\t')
        print(f"entropy_loss={entropy_loss:.6f}", end='\t')
        print(f"total_loss={total_loss:.6f}", end='\t')
        print(f"lr={lr:.6f}", end='\t')
        print(f"entropy_beta={entropy_beta:.6f}", end='\t')
        print()

        # 保存模型
        if (episode + 1) % 500 == 0:
            export_onnx(agent.model, filename=f"mario-a2c-1-episode-{episode+1}.onnx")

        # update entropy beta
        agent.entropy_scheduler.step()

    env.close()
    return agent.model


def test(policy_net):
    print("\nStarting test (10 episodes)...")
    env = JoypadSpace(gym_super_mario_bros.make(GAME_NAME, max_episode_steps=MAX_EPISODE_STEPS), SIMPLE_MOVEMENT)
    frame_stacker = FrameStacker(FRAME_STACK)

    for i in range(10):
        state, _ = env.reset()
        state = frame_stacker.reset(preprocess_state(state))
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                _, logits = policy_net(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                # action = logits.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            state = frame_stacker.step(preprocess_state(state))
            done = terminated or truncated
            total_reward += float(reward)
        print(f"Test Episode {i}: Reward = {total_reward}")
    env.close()



if __name__ == "__main__":
    trained_net = train()
    test(trained_net)
    export_onnx(trained_net)
