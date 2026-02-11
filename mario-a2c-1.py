import ale_py
import time
import torch
import numpy as np
import gymnasium as gym
import cv2

from collections import deque
from rich import print

# Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
FRAME_STACK = 4
LEARNING_RATE = 5e-4
TRAIN_EPISODES = 3000
GRADIENT_CLIP = 1.0
VALUE_LOSS_COEF = 0.5
ADV_NORM_EPS = 1e-8
ENTROPY_BETA_START = 0.05
ENTROPY_BETA_END = 0.01
ENTROPY_BETA_DECAY = 0.999
BATCH_SIZE = 512

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state(state):
    # 下采样 210×160 → 84×84，减少计算量，去除无用信息
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return (state.astype(np.float32) / 127.5) - 1.0


class FrameStacker:
    """堆叠连续帧以捕获时序信息（运动方向、速度等）"""

    def __init__(self, k: int = FRAME_STACK):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        """重置时用同一帧填充"""
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)  # [k, H, W]

    def step(self, frame):
        """添加新帧，返回堆叠结果"""
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)  # [k, H, W]


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


class ActorCriticNet(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super(ActorCriticNet, self).__init__()
        # 输入通道数现在是 FRAME_STACK (4) 而不是 1
        in_channels = FRAME_STACK
        self.conv_net = torch.nn.Sequential(
            # Nature DQN 标准架构 (DeepMind 2015)
            # Input: 4×84×84
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            # 32×20×20
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            # 64×9×9
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            # 64×7×7
        )
        self.feature_net = torch.nn.Sequential(
            self.conv_net,
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64 * 7 * 7, 512),
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

    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_net(features)
        logits = self.policy_net(features)
        return value, logits

    def action_probs(self, x):
        features = self.feature_net(x)
        return torch.nn.functional.softmax(self.policy_net(features), dim=-1)


class Agent:
    def __init__(self, state_shape, action_dim):
        self.model = ActorCriticNet(state_shape, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TRAIN_EPISODES, eta_min=LEARNING_RATE * 0.1
        )
        self.entropy_scheduler = EntropyScheduler(ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY)

        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.next_states = []
        self.terminateds = []

    def select_action(self, state):
        # state 已经是 [FRAME_STACK, H, W] 的堆叠帧
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)  # [1, FRAME_STACK, H, W]
        # print(f"state_t.shape={state_t.shape}")
        value, logits = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()

    def store_transition(self, log_prob, value, reward, entropy, next_state, terminated):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        self.next_states.append(torch.FloatTensor(next_state).to(device))
        self.terminateds.append(torch.tensor([float(terminated)], dtype=torch.float, device=device))

    def update_batch(self):
        if len(self.rewards) == 0:
            return

        log_probs = torch.cat(self.log_probs)  # [B]
        values = torch.cat(self.values).squeeze(-1)  # [B]
        entropies = torch.cat(self.entropies)  # [B]
        rewards = torch.cat(self.rewards).squeeze(-1)  # [B]
        next_states = torch.stack(self.next_states)  # [B, FRAME_STACK, H, W]
        terminateds = torch.cat(self.terminateds).squeeze(-1)  # [B] (1.0 if terminated else 0.0)

        with torch.no_grad():
            next_values, _ = self.model(next_states)
            next_values = next_values.squeeze(-1)  # [B]

            # GAE (Generalized Advantage Estimation) 计算
            # 比 1-step TD 有更低的方差，更稳定的训练
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                # delta = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
                delta = rewards[t] + GAMMA * next_values[t] * (1.0 - terminateds[t]) - values[t].detach()
                # GAE: A_t = δ_t + (γλ) * (1 - done) * A_{t+1}
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - terminateds[t]) * gae
                advantages[t] = gae

            # targets for value function: V_target = A + V
            targets = advantages + values.detach()

        adv_detached = advantages.detach()
        adv_norm = (adv_detached - adv_detached.mean()) / (adv_detached.std(unbiased=False) + ADV_NORM_EPS)

        entropy_beta = self.entropy_scheduler.get_beta()
        critic_loss = torch.nn.functional.mse_loss(values, targets)
        actor_loss = -(log_probs * adv_norm).mean()
        entropy_loss = -entropies.mean() * entropy_beta
        loss = actor_loss + VALUE_LOSS_COEF * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

        self.clear_memory()

    def clear_memory(self):
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.next_states = []
        self.terminateds = []

    def step_scheduler(self):
        current_lr = self.lr_scheduler.get_last_lr()[0]
        if current_lr > self.lr_scheduler.eta_min:
            self.lr_scheduler.step()
        return current_lr


def train(episodes: int = TRAIN_EPISODES, max_steps: int = 4096):
    env = gym.make(
        "ALE/MarioBros-v5",
        full_action_space=False,
        obs_type="grayscale",
        frameskip=4,
    )
    print(f"env.action_space.n={env.action_space.n}")
    print(f"env.observation_space.shape={env.observation_space.shape}")
    agent = Agent(env.observation_space.shape, int(env.action_space.n))
    hist_rewards = deque(maxlen=100)
    frame_stacker = FrameStacker(FRAME_STACK)

    for episode in range(episodes):
        raw_state, _ = env.reset()
        state = frame_stacker.reset(normalize_state(raw_state))  # [FRAME_STACK, H, W]
        total_reward = 0.0
        steps = 0
        done = False
        total_entropy = 0.0

        while not done:
            action, log_prob, value, entropy = agent.select_action(state)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            next_state = frame_stacker.step(normalize_state(next_raw_state))  # [FRAME_STACK, H, W]
            reward /= 800.0
            agent.store_transition(
                log_prob,
                value,
                reward,
                entropy,
                next_state,
                bool(terminated),
            )
            if len(agent.rewards) >= BATCH_SIZE:
                agent.update_batch()

            state = next_state
            total_reward += reward
            total_entropy += float(entropy.item())
            steps += 1
            done = terminated or truncated or steps >= max_steps

        if len(agent.rewards) > 0:
            agent.update_batch()
        agent.entropy_scheduler.step()
        lr = agent.step_scheduler()

        hist_rewards.append(total_reward)
        mean_reward = sum(hist_rewards) / len(hist_rewards)
        entropy_beta = agent.entropy_scheduler.get_beta()
        now = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        print(ts, end="\t")
        print(f"episode={episode}", end="\t")
        print(f"reward={total_reward:+.3f}", end="\t")
        print(f"reward.mean={mean_reward:+.3f}", end="\t")
        print(f"entropy.mean={total_entropy / steps:.3f}", end="\t")
        print(f"steps={steps}", end="\t")
        print(f"lr={lr:.6f}", end="\t")
        print(f"beta={entropy_beta:.6f}", end="\t")
        print()
    env.close()
    return agent.model


def test(policy_net, episodes: int = 1):
    print(f"\nStarting test ({episodes} episodes)...")
    env = gym.make(
        "ALE/MarioBros-v5",
        full_action_space=False,
        obs_type="grayscale",
        frameskip=4,
        render_mode="human",
    )
    frame_stacker = FrameStacker(FRAME_STACK)

    for i in range(episodes):
        raw_state, _ = env.reset()
        state = frame_stacker.reset(normalize_state(raw_state))
        total_reward = 0.0
        total_steps = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, FRAME_STACK, H, W]
                _, logits = policy_net(state_t)
                logits = np.squeeze(logits.cpu().numpy(), axis=0)
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                action = np.random.choice(len(probs), p=probs)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            state = frame_stacker.step(normalize_state(next_raw_state))
            done = terminated or truncated
            total_reward += float(reward)
            total_steps += 1
        print(f"test.episode={i}: reward={total_reward}, steps={total_steps}")
    env.close()


def export_onnx(policy_net, filename="mario-a2c-1.onnx"):
    policy_net.eval()
    dummy_input = torch.randn(1, FRAME_STACK, 84, 84).to(device)  # [1, FRAME_STACK, 84, 84]
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
    print(f"\nModel exported to {filename}")


if __name__ == "__main__":
    trained_net = train()
    test(trained_net)
    export_onnx(trained_net)
