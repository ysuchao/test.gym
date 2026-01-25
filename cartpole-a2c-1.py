import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import cast
from collections import deque
from rich import print

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
TRAIN_EPISODES = 5000
GRADIENT_CLIP = 0.1
VALUE_LOSS_COEF = 0.5
ADV_NORM_EPS = 1e-8
ENTROPY_BETA_START = 0.05
ENTROPY_BETA_END = 0.001
ENTROPY_BETA_DECAY = 0.99
ENTROPY_WARMUP_EPISODES = 200
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_mean = np.array([0, 0, 0, 0])
state_std = np.array([1.0, 1.0, 0.2, 0.3])


def normalize_state(state):
    result = (state - state_mean) / state_std
    return result.astype(np.float32)


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


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_net(features)
        logits = self.policy_net(features)
        return value, logits

    def action_probs(self, x):
        features = self.feature_net(x)
        return F.softmax(self.policy_net(features), dim=-1)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCriticNet(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        self.entropy_scheduler = EntropyScheduler(ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY)

        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.next_states = []
        self.terminateds = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
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
        next_states = torch.stack(self.next_states)  # [B, state_dim]
        terminateds = torch.cat(self.terminateds).squeeze(-1)  # [B] (1.0 if terminated else 0.0)

        with torch.no_grad():
            next_values, _ = self.model(next_states)
            next_values = next_values.squeeze(-1)  # [B]
            targets = rewards + GAMMA * next_values * (1.0 - terminateds)

        advantages = targets - values
        adv_detached = advantages.detach()
        adv_norm = (adv_detached - adv_detached.mean()) / (adv_detached.std(unbiased=False) + ADV_NORM_EPS)

        entropy_beta = self.entropy_scheduler.get_beta()
        critic_loss = F.mse_loss(values, targets)
        actor_loss = -(log_probs * adv_norm).mean()
        entropy_loss = -entropies.mean() * entropy_beta
        loss = actor_loss + VALUE_LOSS_COEF * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
        # print(f"total_norm={total_norm}")
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
        current_lr = self.scheduler.get_last_lr()[0]
        if current_lr > self.scheduler.eta_min:
            self.scheduler.step()
        return current_lr


def train(episodes=TRAIN_EPISODES):
    env = gym.make("CartPole-v1")
    observation_space = cast(Box, env.observation_space)
    action_space = cast(Discrete[np.int64], env.action_space)
    assert observation_space.shape is not None
    state_dim = int(observation_space.shape[0])
    action_dim = int(action_space.n)
    agent = Agent(state_dim, action_dim)

    recent_scores = deque(maxlen=100)

    for episode in range(episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            shaped_reward = float(reward)
            # shaped_reward += np.sum(np.cos(np.clip(next_state, -np.pi, np.pi))) * 0.25
            next_state_norm = normalize_state(next_state)

            agent.store_transition(
                log_prob,
                value,
                shaped_reward,
                entropy,
                next_state_norm,
                bool(terminated),
            )

            if len(agent.rewards) >= BATCH_SIZE:
                agent.update_batch()

            state = next_state_norm
            total_reward += float(reward)
            steps += 1

        if episode >= ENTROPY_WARMUP_EPISODES:
            agent.entropy_scheduler.step()
        lr = agent.step_scheduler()

        if len(agent.rewards) > 0:
            agent.update_batch()

        recent_scores.append(total_reward)
        mean_score = sum(recent_scores) / len(recent_scores)
        entropy_beta = agent.entropy_scheduler.get_beta()
        now = time.time()
        ts = time.strftime('%H:%M:%S', time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        print(ts, end='\t')
        print(f"episode={episode}", end='\t')
        print(f"reward={total_reward:.1f}", end='\t')
        print(f"mean_score={mean_score:.1f}", end='\t')
        print(f"lr={lr:.6f}", end='\t')
        print(f"entropy_beta={entropy_beta:.6f}", end='\t')
        print()

        if mean_score >= 500:
            print(f"Solved! Training finished at episode {episode}. Avg Score: {mean_score}")
            break

    env.close()
    return agent.model


def test(policy_net):
    print("\nStarting test (10 episodes)...")
    env = gym.make("CartPole-v1")

    for i in range(10):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                state = normalize_state(state)
                state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
                _, logits = policy_net(state_t)
                action = logits.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
        print(f"Test Episode {i}: Reward = {total_reward}")
    env.close()


def export_onnx(policy_net, filename="cartpole-a2c-1.onnx"):
    policy_net.eval()
    dummy_input = torch.randn(1, 4).to(device)
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
