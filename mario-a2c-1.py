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
LEARNING_RATE = 5e-4
TRAIN_EPISODES = 3000
GRADIENT_CLIP = 1.0
VALUE_LOSS_COEF = 1
ADV_NORM_EPS = 1e-8
ENTROPY_BETA_START = 0.001
ENTROPY_BETA_END = 0.0001
ENTROPY_BETA_DECAY = 0.999
BATCH_SIZE = 512

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state(state):
    return (state.astype(np.float32) / 127.5) - 1.0


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
        self.conv_net = torch.nn.Sequential(
            # Input: 1×210×160
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # 32×210×160
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # 64×105×80
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # 128×53×40
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # 64×27×20
        )
        self.feature_net = torch.nn.Sequential(
            self.conv_net,
            torch.nn.AdaptiveAvgPool2d((9, 5)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(64 * 9 * 5, 512),
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
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0)
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
        next_states = torch.stack(self.next_states).unsqueeze(1)  # [B, state_dim]
        terminateds = torch.cat(self.terminateds).squeeze(-1)  # [B] (1.0 if terminated else 0.0)

        with torch.no_grad():
            # print(f"next_states.shape={next_states.shape}")
            next_values, _ = self.model(next_states)
            next_values = next_values.squeeze(-1)  # [B]
            targets = rewards + GAMMA * next_values * (1.0 - terminateds)

        advantages = targets - values
        # print(f"advantages={advantages.cpu().numpy()}", end="\t")
        adv_detached = advantages.detach()
        adv_norm = (adv_detached - adv_detached.mean()) / (adv_detached.std(unbiased=False) + ADV_NORM_EPS)

        entropy_beta = self.entropy_scheduler.get_beta()
        critic_loss = torch.nn.functional.mse_loss(values, targets.detach())
        # log_probs_cpu = log_probs.cpu().detach()
        # adv_norm_cpu = adv_norm.cpu().detach()
        # print(f"log_probs={log_probs_cpu.numpy()}", end="\t")
        # print(f"adv_norm={adv_norm_cpu.numpy()}", end="\t")
        actor_loss = -(log_probs * adv_norm).mean()
        entropy_loss = -entropies.mean() * entropy_beta
        # print(f"actor_loss={actor_loss.item():.3f}", end="\t")
        # print(f"critic_loss={critic_loss.item():.3f}", end="\t")
        # print(f"entropy_loss={entropy_loss.item():.3f}", end="\t")
        # print(f"beta={entropy_beta:.6f}", end="\t")
        loss = actor_loss + VALUE_LOSS_COEF * critic_loss + entropy_loss
        # loss_cpu = loss.cpu().detach()
        # print(f"total_loss={loss_cpu.numpy():.3f}", end="\t")
        # print()

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

    for episode in range(episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0.0
        steps = 0
        done = False
        total_entropy = 0.0

        while not done:
            action, log_prob, value, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_norm = normalize_state(next_state)
            reward /= 800.0
            agent.store_transition(
                log_prob,
                value,
                reward,
                entropy,
                next_state_norm,
                bool(terminated),
            )
            if len(agent.rewards) >= BATCH_SIZE:
                agent.update_batch()

            state = next_state_norm
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

    for i in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        total_steps = 0
        done = False
        while not done:
            state = normalize_state(state)
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
                # print(f"state.shape={state.shape}")
                _, logits = policy_net(state)
                # assert False, logits
                logits = np.squeeze(logits.cpu().numpy(), axis=0)
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                action = np.random.choice(len(probs), p=probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            total_steps += 1
        print(f"test.episode={i}: reward={total_reward}, steps={total_steps}")
    env.close()


def export_onnx(policy_net, filename="mario-a2c-1.onnx"):
    policy_net.eval()
    dummy_input = torch.randn(1, 1, 210, 160).to(device)
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
