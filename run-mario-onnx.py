import sys
import ale_py
import gymnasium as gym
import numpy as np
import onnxruntime as ort
import cv2

from collections import deque


# 必须和训练时一致
FRAME_STACK = 4
gym.register_envs(ale_py)


def normalize_state(state: np.ndarray) -> np.ndarray:
    # 下采样 210×160 → 84×84，与训练预处理一致
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return (state.astype(np.float32) / 127.5) - 1.0


def run_onnx_model(
    model_path: str,
    num_episodes: int = 5,
    full_action_space: bool = False,  # 必须和训练一致（训练时 full_action_space=False）
    frameskip: int = 4,
    render: bool = True,
) -> None:
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = next(iter(session.get_inputs())).name
    output_names = [o.name for o in session.get_outputs()]

    env = gym.make(
        "ALE/MarioBros-v5",
        full_action_space=full_action_space,
        obs_type="grayscale",
        frameskip=frameskip,
        render_mode="human" if render else None,
    )

    # 用 deque 管理帧堆叠，与训练时 FrameStacker 逻辑一致
    frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)

    for ep in range(num_episodes):
        state_raw, _ = env.reset()
        state = normalize_state(np.asarray(state_raw))
        # 重置时用同一帧填充（与训练一致）
        frames.clear()
        for _ in range(FRAME_STACK):
            frames.append(state)

        total_reward = 0.0
        done = False
        step_count = 0
        while not done:
            # [FRAME_STACK, 84, 84] → [1, FRAME_STACK, 84, 84]
            input_data = np.expand_dims(np.stack(frames, axis=0).astype(np.float32), axis=0)
            results = session.run(output_names, {input_name: input_data})
            # results[1] 是 action logits（不是 probs，ONNX 导出的是 raw logits）
            logits = np.squeeze(np.asarray(results[1]), axis=0)
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            action = int(np.argmax(probs))

            # 打印每种 action 的概率
            action_probs_str = " | ".join([f"A{a}:{p:.3f}" for a, p in enumerate(probs)])
            print(f"step={step_count:4d} | {action_probs_str} | chose=A{action}")

            state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = normalize_state(np.asarray(state_raw))
            frames.append(state)  # deque(maxlen=FRAME_STACK) 自动丢弃最旧帧
            step_count += 1
        print(f"test.episode={ep}, reward={total_reward:.3f}, steps={step_count}")
    env.close()


if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "mario-a2c-1.onnx"
    try:
        run_onnx_model(model_file)
    except Exception as e:
        print(f"error: {e}")
