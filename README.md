# MNIST Demo

基于 PyTorch 的 MNIST 手写数字识别项目。

## 项目简介

本项目实现了一个简单的三层神经网络（784 -> 128 -> 64 -> 10），用于 MNIST 手写数字分类任务。

## 环境要求

- Python 3.11+
- uv 包管理器（0.11.3+）

## 安装

```bash
uv sync
```

## 运行

```bash
uv run python main.py
```

训练过程会实时显示：
- 训练损失曲线
- 训练准确率和测试准确率曲线
- 模型预测示例

## 项目结构

```
mnist-demo/
├── layers.py      # 神经网络模型定义
├── main.py        # 训练、评估和可视化
├── pyproject.toml # 项目配置和依赖
└── AGENTS.md      # AI 代理开发指南
```

## 模型架构

- 输入层：28x28 图像展平为 784 维
- 隐藏层 1：128 个神经元 + ReLU 激活
- 隐藏层 2：64 个神经元 + ReLU 激活
- 输出层：10 个神经元（数字 0-9）

## 超参数

- Batch Size: 512
- Learning Rate: 0.001 (Adam)
- Epochs: 15
