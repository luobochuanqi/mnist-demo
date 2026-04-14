import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from layers import SimpleNN

BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


def show_sample_images(dataset, num=6):
    """显示前几张图片和它们的标签"""
    fig, axes = plt.subplots(1, num, figsize=(10, 3))
    for i in range(num):
        img, label = dataset[i]
        # img 形状是 [1, 28, 28]，squeeze(0) 去掉通道维，变成 28x28
        axes[i].imshow(img.squeeze(0), cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    plt.show()


def evaluate(model, loader):
    """测试函数，用来评估当前模型准确率"""
    # 切换到评估模式（会关闭 dropout 等）
    model.eval()
    correct = 0
    total = 0
    # 测试时不计算梯度，节省显存
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def main():
    print("Hello from mnist-demo!")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练样本数：{len(train_dataset)}")
    print(f"测试样本数：{len(test_dataset)}")
    print(f"当前运行设备: {DEVICE}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print("PyTorch CUDA版本：", torch.version.cuda)
    # 观察数据
    # show_sample_images(train_dataset)

    model = SimpleNN().to(DEVICE)
    # 观察神经网络
    print(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accs = []
    test_accs = []

    plt.ion()  # 打开交互绘图模式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 切换回训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 计算本 epoch 指标
        avg_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

        # ---- 画图 ----
        ax1.clear()
        ax2.clear()

        ax1.plot(range(1, epoch + 1), train_losses, "b-o", label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(range(1, epoch + 1), train_accs, "g-o", label="Train Acc")
        ax2.plot(range(1, epoch + 1), test_accs, "r-s", label="Test Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.pause(0.1)  # 短暂暂停，刷新图像

    plt.ioff()  # 关闭交互模式
    plt.show()  # 最后保持图像显示


if __name__ == "__main__":
    main()
