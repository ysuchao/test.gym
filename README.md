# Gymnasium demo

这是一个简单的 Gymnasium 示例项目，演示如何使用 `CartPole-v1` 环境运行一个随机代理（random-agent）。

依赖（示例）:

- python 3.8+
- gymnasium
- numpy

安装并运行：

```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 运行示例
python main.py
```

示例会运行若干个 episode，并在终端打印每一集的累计 reward 与步数。该示例不需要显示器（headless 环境可运行）。
