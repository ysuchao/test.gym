import sys
import gymnasium as gym
import numpy as np
import onnxruntime as ort

state_mean = [0, 0, 0, 0]
state_std = [1, 1, 0.2, 0.3]


# 关闭科学计数法显示
np.set_printoptions(suppress=True)


def normalize_state(state):
    result = np.array(
        [(state[i] - state_mean[i]) / state_std[i] for i in range(len(state))],
        dtype=np.float32,
    )
    return result


def run_onnx_model(model_path, num_episodes=10):
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # Create environment with visualization
    env = gym.make("CartPole-v1", render_mode="human")

    for i in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Prepare input for ONNX runtime
            # Observation is (4,) -> needs to be (1, 4) float32
            state = normalize_state(state)
            input_data = state.astype(np.float32).reshape(1, 4)

            # Run inference
            outputs = session.run(None, {input_name: input_data})
            action = np.argmax(outputs[-1])

            # Step environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"test.episode={i}, reward={total_reward}")
    env.close()


if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "cartpole-double-dqn.onnx"
    try:
        run_onnx_model(model_file)
    except Exception as e:
        print(f"error: {e}")
