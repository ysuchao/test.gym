import time
import random
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from rich import print

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 512
EPSILON_START = 0.1
EPSILON_END = 0.001
EPSILON_DECAY = 0.995
TRAIN_EPISODES = 100000
GRADIENT_CLIP = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_mean = [0, 0, 0, 0]
state_std = [1, 1, 0.2, 0.3]
def normalize_state(state):
    result = np.array([(state[i] - state_mean[i]) / state_std[i] for i in range(len(state))], dtype=np.float32)
    return result

class EpsilonScheduler:
    def __init__(self, start, end, decay):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def step(self):
        self.epsilon = max(self.end, self.epsilon * self.decay)
        return self.epsilon

    def get_epsilon(self):
        return self.epsilon


class DQN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon_scheduler = EpsilonScheduler(EPSILON_START, EPSILON_END, EPSILON_DECAY)

    def select_action(self, state):
        if random.random() < self.epsilon_scheduler.get_epsilon():
            return random.randrange(self.action_dim)
        return self.select_greedy_action(state)

    def select_greedy_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
            return self.policy_net(state_t).argmax().item()

    def sync(self, tau: float = 0.005):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        b_state, b_action, b_reward, b_next_state, b_done = self.memory.sample(BATCH_SIZE)
        b_state = torch.FloatTensor(b_state).to(device)
        b_action = torch.LongTensor(b_action).to(device).unsqueeze(1)
        b_reward = torch.FloatTensor(b_reward).to(device).unsqueeze(1)
        b_next_state = torch.FloatTensor(b_next_state).to(device)
        b_done = torch.FloatTensor(b_done).to(device).unsqueeze(1)

        # Double DQN update
        current_q = self.policy_net(b_state).gather(1, b_action)
        with torch.no_grad():
            next_actions = self.policy_net(b_next_state).argmax(1).unsqueeze(1)
            next_q = self.target_net(b_next_state).gather(1, next_actions)
            expected_q = b_reward + (1 - b_done) * GAMMA * next_q

        # Huber Loss (SmoothL1Loss) - 稳定性提升
        loss = torch.nn.SmoothL1Loss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=GRADIENT_CLIP)
        self.optimizer.step()


def train(episodes: int = TRAIN_EPISODES):
    env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    latest_steps = deque(maxlen=100)
    good = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0
        total_steps = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward -= np.abs(next_state).sum() * 0.5
            next_state = normalize_state(next_state)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_steps += 1
            agent.update()
            agent.sync()

        # 学习率调度器在优化器step之后调用（PyTorch最佳实践）
        current_lr = agent.optimizer_scheduler.get_last_lr()[0]
        if current_lr > agent.optimizer_scheduler.eta_min:
            agent.optimizer_scheduler.step()

        latest_steps.append(total_steps)
        avg_steps = sum(latest_steps) / len(latest_steps)
        min_steps = min(latest_steps)
        good = (good + 1) * (total_steps >= 500)

        print(f"{time.strftime('%H:%M:%S', time.localtime())}.{int(time.time() * 1000) % 1000:03d}", end="\t")
        print(f"episode={episode}", end="\t")
        print(f"epsilon={agent.epsilon_scheduler.get_epsilon():.6f}", end="\t")
        print(f"lr={current_lr:.6f}", end="\t")
        print(f"steps.new={total_steps:.3f}", end="\t")
        print(f"steps.min={min_steps:.3f}", end="\t")
        print(f"steps.avg={avg_steps:.3f}", end="\t")
        print(f"reward={total_reward:.3f}", end="\t")
        print(f"good={good}", end="\t")
        print()

        # early exit
        if good >= len(latest_steps):
            print(f"solved -- training finished at episode {episode+1}")
            break
    env.close()
    return agent.policy_net


def test(policy_net):
    print("starting test (10 times)...")
    env = gym.make("CartPole-v1")
    for i in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state = normalize_state(state)
                state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
                action = policy_net(state_t).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"test.episode={i}, reward={total_reward}")
    env.close()


def export_onnx(policy_net, filename="cartpole-double-dqn.onnx"):
    policy_net.eval()
    dummy_input = torch.randn(1, 4).to(device)
    torch.onnx.export(
        policy_net,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"\nModel exported to {filename}")


if __name__ == "__main__":
    trained_net = train()
    test(trained_net)
    export_onnx(trained_net)
