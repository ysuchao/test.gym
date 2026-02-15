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
BATCH_SIZE = 1024

# RND (Random Network Distillation) 超参数
# ==========================================
# RND 通过"好奇心"解决稀疏 reward 问题：
#   - target 网络: 随机初始化后冻结，输出固定的随机特征
#   - predictor 网络: 可训练，尝试预测 target 的输出
#   - intrinsic reward = 预测误差
#   - 见过的状态 → 预测准确 → 低 reward（"无聊"）
#   - 新状态 → 预测不准 → 高 reward（"好奇！"）
RND_OUTPUT_DIM = 128       # RND 嵌入维度（target/predictor 的输出大小）
RND_LEARNING_RATE = 1e-4   # predictor 的学习率（比主网络低，避免学太快失去好奇心）
RND_COEF = 1.0             # intrinsic reward 的混合系数
GAMMA_INT = 0.99           # intrinsic reward 的折扣因子（当前与 GAMMA 相同；如需更短视的探索可调低至 0.95）

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


class RunningMeanStd:
    """Welford 在线算法，追踪 mean 和 variance

    用于归一化 RND 的 intrinsic reward：
    - RND 的预测误差（prediction error）的绝对值会随训练变化
    - 训练初期误差大（一切都新奇），后期误差小（大部分都见过）
    - 不归一化的话，intrinsic reward 的 scale 不稳定，GAE 计算会出问题
    - 用 running mean/std 归一化后，intrinsic reward 始终在合理范围内

    Welford 算法优势：
    - 数值稳定（不会因大量累加产生浮点误差）
    - O(1) 空间和时间（不需要存历史数据）
    - 在线更新（每个 batch 增量更新）
    """

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # 避免除零，同时给 var 一个初始"先验"

    def update(self, x):
        """用一个 batch 的数据增量更新 mean 和 var

        Args:
            x: numpy array，一个 batch 的 intrinsic reward 值
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """并行 Welford 合并公式

        将两组统计量（已有的 + 新 batch 的）合并为一组：
        - 新 mean = 加权平均
        - 新 var = 合并方差公式（考虑两组 mean 的差异）
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        # M2 = sum of squared differences from mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    # 注意：归一化 intrinsic reward 时，只除以 std，不减 mean
    # 原因：RND paper 中 intrinsic reward 始终为正（MSE），减去 mean 会产生负 reward
    # 实际归一化在训练循环中直接做：reward_int_raw / (sqrt(var) + eps)


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


class RNDModel(torch.nn.Module):
    """Random Network Distillation (RND) — 好奇心驱动探索

    核心思想：
    - target 网络：随机初始化后永远冻结，对同一个输入永远输出相同的随机特征
    - predictor 网络：可训练，尝试模仿 target 的输出
    - prediction error = intrinsic reward（内在奖励）

    为什么有效？
    - 见过很多次的状态 → predictor 学会了预测 → error 低 → reward 低（"无聊"）
    - 从未见过的状态 → predictor 还没学会 → error 高 → reward 高（"好奇！"）
    - 这样 agent 被鼓励去探索新状态，而不是反复走老路

    为什么用单帧而不是堆叠帧？
    - RND 判断的是"这个画面是否新奇"，不需要运动信息
    - 用单帧可以减少参数量，加快训练
    - 堆叠帧的相同画面在不同位置出现时，RND 会认为是不同状态（不好）
    """

    def __init__(self, input_shape=(1, 84, 84), output_dim=RND_OUTPUT_DIM):
        super(RNDModel, self).__init__()

        # Target 网络：随机初始化，永远不训练
        # 它的作用是提供一个固定的"随机映射"，把图像映射到一个随机空间
        self.target = torch.nn.Sequential(
            # 使用比 Nature DQN 更小的网络（只需要判断新奇度，不需要决策）
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),   # 1×84×84 → 32×20×20
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 64×9×9
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 64×7×7
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, output_dim),           # → RND_OUTPUT_DIM
        )

        # Predictor 网络：可训练，尝试模仿 target 的输出
        # 比 target 多一层 FC，给它更多容量来学习映射
        # （如果 predictor 和 target 完全一样，可能太容易学会 → 好奇心消失太快）
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 512),                  # 多一层 FC
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim),                   # → RND_OUTPUT_DIM
        )

        # 冻结 target 网络的所有参数
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        """计算 target 和 predictor 的输出

        Args:
            obs: [B, 1, 84, 84] 单帧灰度图（归一化后）

        Returns:
            target_features: [B, RND_OUTPUT_DIM] target 网络的输出（固定不变）
            predictor_features: [B, RND_OUTPUT_DIM] predictor 的预测
        """
        target_features = self.target(obs)
        predictor_features = self.predictor(obs)
        return target_features, predictor_features

    def compute_intrinsic_reward(self, obs):
        """计算 intrinsic reward = prediction error

        Args:
            obs: [B, 1, 84, 84] 单帧灰度图

        Returns:
            intrinsic_rewards: [B] 每个样本的内在奖励（MSE per sample）
        """
        with torch.no_grad():
            target_features, predictor_features = self.forward(obs)
            # 每个样本的 MSE（不 reduce 到 scalar，保留 batch 维度）
            intrinsic_rewards = (target_features - predictor_features).pow(2).mean(dim=1)
        return intrinsic_rewards


class Agent:
    def __init__(self, state_shape, action_dim):
        self.model = ActorCriticNet(state_shape, action_dim).to(device)

        # Intrinsic value head：独立的 value 网络，专门估算 intrinsic reward 的价值
        # 为什么需要两个 value head？
        # - extrinsic reward（游戏分数）和 intrinsic reward（好奇心）性质完全不同
        # - extrinsic：稀疏、scale 固定（0 或 800 的倍数）
        # - intrinsic：密集、scale 随训练变化（predictor 学习后会衰减）
        # - 用同一个 value head 估算两种不同性质的 reward 会互相干扰
        # - 分开后各自可以用不同的 discount factor（GAMMA vs GAMMA_INT）
        self.intrinsic_value_net = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        ).to(device)

        # RND 网络
        self.rnd = RNDModel().to(device)

        # 主优化器：actor-critic + intrinsic value head
        # intrinsic_value_net 的参数也加进来一起优化
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.intrinsic_value_net.parameters()),
            lr=LEARNING_RATE,
        )
        # RND predictor 单独的优化器（学习率更低，避免 predictor 学太快导致好奇心过早消失）
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(), lr=RND_LEARNING_RATE
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TRAIN_EPISODES, eta_min=LEARNING_RATE * 0.1
        )
        self.entropy_scheduler = EntropyScheduler(ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY)

        # Intrinsic reward 归一化器
        self.rnd_reward_rms = RunningMeanStd()

        self.log_probs = []
        self.values = []
        self.intrinsic_values = []  # 新增：intrinsic value 估算
        self.entropies = []
        self.rewards_ext = []       # 重命名：extrinsic rewards
        self.rewards_int = []       # 新增：intrinsic rewards
        self.states = []            # 新增：当前状态（用于 intrinsic value 梯度计算）
        self.next_states = []
        self.terminateds = []

    def select_action(self, state):
        # state 已经是 [FRAME_STACK, H, W] 的堆叠帧
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)  # [1, FRAME_STACK, H, W]
        value, logits = self.model(state_t)

        # 计算 intrinsic value（不需要梯度，GAE 中只用 detach 后的值）
        # 梯度在 update_batch 中通过 intrinsic_values_recomputed 重新计算
        with torch.no_grad():
            features = self.model.feature_net(state_t)
            intrinsic_value = self.intrinsic_value_net(features)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, intrinsic_value, dist.entropy()

    def compute_intrinsic_reward(self, next_state):
        """用 RND 计算当前状态的 intrinsic reward

        Args:
            next_state: [FRAME_STACK, H, W] 堆叠帧

        Returns:
            float: 归一化后的 intrinsic reward
        """
        # RND 只用单帧（最后一帧），不需要堆叠帧
        # next_state shape: [FRAME_STACK, H, W]，取最后一帧 → [1, H, W]
        single_frame = torch.FloatTensor(next_state[-1:]).to(device).unsqueeze(0)  # [1, 1, 84, 84]
        intrinsic_reward = self.rnd.compute_intrinsic_reward(single_frame)
        return intrinsic_reward.item()

    def store_transition(self, log_prob, value, intrinsic_value, reward_ext, reward_int, entropy, state, next_state, terminated):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.intrinsic_values.append(intrinsic_value)
        self.entropies.append(entropy)
        self.rewards_ext.append(torch.tensor([reward_ext], dtype=torch.float, device=device))
        self.rewards_int.append(torch.tensor([reward_int], dtype=torch.float, device=device))
        self.states.append(torch.FloatTensor(state).to(device))
        self.next_states.append(torch.FloatTensor(next_state).to(device))
        self.terminateds.append(torch.tensor([float(terminated)], dtype=torch.float, device=device))

    def update_batch(self):
        if len(self.rewards_ext) == 0:
            return

        log_probs = torch.cat(self.log_probs)                      # [B]
        values = torch.cat(self.values).view(-1)                    # [B]
        intrinsic_values = torch.cat(self.intrinsic_values).view(-1)  # [B]
        entropies = torch.cat(self.entropies)                       # [B]
        rewards_ext = torch.cat(self.rewards_ext).view(-1)          # [B]
        rewards_int = torch.cat(self.rewards_int).view(-1)          # [B]
        next_states = torch.stack(self.next_states)                 # [B, FRAME_STACK, H, W]
        terminateds = torch.cat(self.terminateds).view(-1)          # [B]

        # ========== RND predictor 更新 ==========
        # 用 batch 中的 next_states 训练 predictor 网络
        # 注意：RND 用单帧，取堆叠帧的最后一帧
        single_frames = next_states[:, -1:, :, :]  # [B, 1, 84, 84]
        target_features, predictor_features = self.rnd(single_frames)
        # predictor loss = MSE(predictor_output, target_output)
        # target 已经冻结，所以梯度只流向 predictor
        rnd_loss = torch.nn.functional.mse_loss(predictor_features, target_features.detach())

        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        # ========== 双 GAE 计算 ==========
        # 分别为 extrinsic 和 intrinsic reward 计算 GAE
        # 为什么要分开？
        # - extrinsic reward 在 episode 结束时需要截断（terminated → V=0）
        # - intrinsic reward 不应该被 episode 边界截断！
        #   因为好奇心是"永恒"的：即使 Mario 死了，对未探索区域的好奇心不应重置
        #   所以 intrinsic GAE 忽略 terminated flag

        with torch.no_grad():
            next_values_ext, _ = self.model(next_states)
            next_values_ext = next_values_ext.squeeze(-1)  # [B]

            next_features = self.model.feature_net(next_states)
            next_values_int = self.intrinsic_value_net(next_features).squeeze(-1)  # [B]

            # Extrinsic GAE（正常的，考虑 terminated）
            advantages_ext = torch.zeros_like(rewards_ext)
            gae_ext = 0.0
            for t in reversed(range(len(rewards_ext))):
                delta = rewards_ext[t] + GAMMA * next_values_ext[t] * (1.0 - terminateds[t]) - values[t].detach()
                gae_ext = delta + GAMMA * GAE_LAMBDA * (1.0 - terminateds[t]) * gae_ext
                advantages_ext[t] = gae_ext

            # Intrinsic GAE（不考虑 terminated，好奇心跨 episode 持续）
            advantages_int = torch.zeros_like(rewards_int)
            gae_int = 0.0
            for t in reversed(range(len(rewards_int))):
                # 注意：这里没有 (1 - terminated)！intrinsic reward 不被 episode 边界截断
                delta = rewards_int[t] + GAMMA_INT * next_values_int[t] - intrinsic_values[t].detach()
                gae_int = delta + GAMMA_INT * GAE_LAMBDA * gae_int
                advantages_int[t] = gae_int

            # 合并 advantages：extrinsic + intrinsic
            # 两者都对 actor 有贡献：agent 既追求游戏分数，也追求探索新奇状态
            advantages = advantages_ext + RND_COEF * advantages_int

            # 各自的 value targets
            targets_ext = advantages_ext + values.detach()
            targets_int = advantages_int + intrinsic_values.detach()

        # ========== 主网络更新 ==========
        adv_detached = advantages.detach()
        adv_norm = (adv_detached - adv_detached.mean()) / (adv_detached.std(unbiased=False) + ADV_NORM_EPS)

        entropy_beta = self.entropy_scheduler.get_beta()

        # Extrinsic critic loss
        critic_loss_ext = torch.nn.functional.mse_loss(values, targets_ext)
        # Intrinsic critic loss（训练 intrinsic value head）
        # 需要重新前向计算 intrinsic_values（因为需要梯度）
        # 用存储的当前状态 s_t 来计算 V_int(s_t)
        states = torch.stack(self.states)  # [B, FRAME_STACK, H, W]
        # .detach() 防止梯度通过 feature_net 传播第二次
        # 没有 detach 的话，loss.backward() 会经过 feature_net 两条路径：
        #   1) actor/critic 的正常前向 → 正确的梯度
        #   2) 这里的 intrinsic value 重计算 → 额外的、不一致的梯度
        # detach 后，只训练 intrinsic_value_net 的权重，feature_net 只接收来自 actor/critic 的梯度
        intrinsic_values_recomputed = self.intrinsic_value_net(
            self.model.feature_net(states).detach()
        ).squeeze(-1)
        critic_loss_int = torch.nn.functional.mse_loss(intrinsic_values_recomputed, targets_int)

        actor_loss = -(log_probs * adv_norm).mean()
        entropy_loss = -entropies.mean() * entropy_beta
        loss = actor_loss + VALUE_LOSS_COEF * (critic_loss_ext + critic_loss_int) + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # clip 前计算梯度范数，用于监控梯度健康度（爆炸/消失）
        all_params = list(self.model.parameters()) + list(self.intrinsic_value_net.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, GRADIENT_CLIP)
        self.optimizer.step()

        self.clear_memory()
        # 返回各项 loss 和梯度范数，用于训练监控
        return {
            "rnd_loss": rnd_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss_ext": critic_loss_ext.item(),
            "critic_loss_int": critic_loss_int.item(),
            "grad_norm": grad_norm.item(),
        }

    def clear_memory(self):
        self.log_probs = []
        self.values = []
        self.intrinsic_values = []
        self.entropies = []
        self.rewards_ext = []
        self.rewards_int = []
        self.states = []
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

    # 用于收集一个 batch 内的 intrinsic rewards，batch 结束后归一化
    intrinsic_reward_batch = []

    for episode in range(episodes):
        raw_state, _ = env.reset()
        state = frame_stacker.reset(normalize_state(raw_state))  # [FRAME_STACK, H, W]
        total_reward_ext = 0.0
        total_reward_int = 0.0
        steps = 0
        done = False
        total_entropy = 0.0
        # 训练指标累积器（每个 episode 内可能有多次 update_batch）
        metrics_accum = {"rnd_loss": 0.0, "actor_loss": 0.0, "critic_loss_ext": 0.0, "critic_loss_int": 0.0, "grad_norm": 0.0}
        metrics_count = 0

        while not done:
            action, log_prob, value, intrinsic_value, entropy = agent.select_action(state)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            next_state = frame_stacker.step(normalize_state(next_raw_state))  # [FRAME_STACK, H, W]
            reward_ext = reward / 800.0

            # 计算 intrinsic reward（RND prediction error）
            reward_int_raw = agent.compute_intrinsic_reward(next_state)
            intrinsic_reward_batch.append(reward_int_raw)

            # 归一化 intrinsic reward
            # 每步都更新 RunningMeanStd，用归一化后的值
            reward_int = reward_int_raw / (np.sqrt(agent.rnd_reward_rms.var) + 1e-8)

            agent.store_transition(
                log_prob,
                value,
                intrinsic_value,
                reward_ext,
                reward_int,
                entropy,
                state,
                next_state,
                bool(terminated),
            )
            if len(agent.rewards_ext) >= BATCH_SIZE:
                # 在 update 之前，用这个 batch 的 intrinsic rewards 更新归一化统计量
                if len(intrinsic_reward_batch) > 0:
                    agent.rnd_reward_rms.update(np.array(intrinsic_reward_batch))
                    intrinsic_reward_batch = []
                rnd_loss = agent.update_batch()
                if rnd_loss is not None:
                    for k in metrics_accum:
                        metrics_accum[k] += rnd_loss[k]
                    metrics_count += 1

            state = next_state
            total_reward_ext += reward_ext
            total_reward_int += reward_int
            total_entropy += float(entropy.item())
            steps += 1
            done = terminated or truncated or steps >= max_steps

        if len(agent.rewards_ext) > 0:
            if len(intrinsic_reward_batch) > 0:
                agent.rnd_reward_rms.update(np.array(intrinsic_reward_batch))
                intrinsic_reward_batch = []
            rnd_loss = agent.update_batch()
            if rnd_loss is not None:
                for k in metrics_accum:
                    metrics_accum[k] += rnd_loss[k]
                metrics_count += 1
        agent.entropy_scheduler.step()
        lr = agent.step_scheduler()

        hist_rewards.append(total_reward_ext)
        mean_reward = sum(hist_rewards) / len(hist_rewards)
        entropy_beta = agent.entropy_scheduler.get_beta()
        # 计算各项 loss 的 episode 平均值
        n = max(metrics_count, 1)
        avg = {k: v / n for k, v in metrics_accum.items()}
        now = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        print(ts, end="\t")
        print(f"episode={episode}", end="\t")
        print(f"reward_ext={total_reward_ext:+.3f}", end="\t")
        print(f"reward_int={total_reward_int:+.3f}", end="\t")
        print(f"reward.mean={mean_reward:+.3f}", end="\t")
        print(f"entropy={total_entropy / steps:.3f}", end="\t")
        print(f"actor={avg['actor_loss']:.4f}", end="\t")
        print(f"c_ext={avg['critic_loss_ext']:.4f}", end="\t")
        print(f"c_int={avg['critic_loss_int']:.4f}", end="\t")
        print(f"rnd={avg['rnd_loss']:.6f}", end="\t")
        print(f"grad={avg['grad_norm']:.4f}", end="\t")
        print(f"rnd_std={np.sqrt(agent.rnd_reward_rms.var):.4f}", end="\t")
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
