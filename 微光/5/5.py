# -*- coding: utf-8 -*-
"""
ml-5 卷积神经网络实战：CIFAR-10 十分类
================================================
内容：
  1. 加载 CIFAR-10，按 45000 / 5000 / 10000 划分训练/验证/测试集，随机展示样本
  2. 从零搭建基础 CNN（两层以上卷积 + 激活 + 池化 + 全局平均池化 + 10 类输出）
  3. 训练基础 CNN 与改进 CNN（BatchNorm + Dropout），记录训练/验证损失与准确率
  4. 绘制训练曲线，保存验证集最优模型
  5. 测试集评估：准确率、混淆矩阵、预测正确/错误样本，并分析 cat/dog 混淆
运行环境：Python 3.9 + PyTorch 2.x（CUDA 或 CPU 均可）
运行方式：python train_cnn.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ========================== 基础配置 ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
for d in (FIG_DIR, CKPT_DIR, DATA_ROOT):
    os.makedirs(d, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
SEED = 42

# 部分新显卡与旧版 PyTorch 不兼容，做一次真实自检，失败则回退 CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _DEVICE.type == "cuda":
    try:
        _a = torch.randn(16, 16, device=_DEVICE)
        _b = _a @ _a
        torch.cuda.synchronize()
    except Exception:
        _DEVICE = torch.device("cpu")
DEVICE = _DEVICE

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
# CIFAR-10 官方均值/方差
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================== 1. 数据准备 ==========================
def prepare_data():
    """加载 CIFAR-10，并固定随机划分训练集 45000 / 验证集 5000 / 测试集 10000"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_full = datasets.CIFAR10(root=DATA_ROOT, train=True,
                                  download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_ROOT, train=False,
                                download=True, transform=transform)

    # 固定随机种子划分，保证两个模型看到完全相同的数据
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        train_full, [45000, 5000], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)
    print(f"训练集 {len(train_set)} / 验证集 {len(val_set)} / 测试集 {len(test_set)}")
    return train_full, train_loader, val_loader, test_loader


def show_data_samples(train_full):
    """随机展示 10 个类别各 3 张图片，观察容易区分/混淆的特征"""
    # 用不归一化的 transform 方便显示
    raw = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False,
                           transform=transforms.ToTensor())
    idx_by_class = {c: [] for c in CLASSES}
    for i, (_, y) in enumerate(raw):
        if len(idx_by_class[CLASSES[y]]) < 3:
            idx_by_class[CLASSES[y]].append(i)
        if all(len(v) == 3 for v in idx_by_class.values()):
            break

    fig, axes = plt.subplots(10, 3, figsize=(6, 14))
    for r, cls in enumerate(CLASSES):
        for c, idx in enumerate(idx_by_class[cls]):
            img, _ = raw[idx]
            axes[r][c].imshow(img.permute(1, 2, 0).numpy())
            axes[r][c].axis("off")
        axes[r][0].set_ylabel(cls, fontsize=10)
    fig.suptitle("CIFAR-10 samples (3 per class)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "data_samples.png"), dpi=120)
    plt.close(fig)
    print("已保存数据样本图: figures/data_samples.png")


# ========================== 2. 模型定义 ==========================
class BaseCNN(nn.Module):
    """基础 CNN：卷积+ReLU+池化 ×3 组，全局平均池化 + 线性分类层"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),          # 全局平均池化 -> 128×1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 10),               # 10 类输出
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """改进 CNN：在基础 CNN 上加入 BatchNorm（每层卷积后）+ Dropout(0.5)"""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def show_layer_shapes(model, input_size=(3, 32, 32)):
    """打印各主要层输入/输出形状（用于报告）"""
    x = torch.randn(1, *input_size)
    print(f"{'层':<40}{'输出形状':<20}")
    print("-" * 62)
    for name, layer in model.features.named_children():
        x = layer(x)
        print(f"{name:<40}{str(tuple(x.shape)):<20}")
    for name, layer in model.classifier.named_children():
        x = layer(x)
        print(f"{name:<40}{str(tuple(x.shape)):<20}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ========================== 3. 训练与验证 ==========================
@torch.no_grad()
def evaluate(model, loader):
    """返回 (准确率, 平均损失)"""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def train_model(model, name, train_loader, val_loader):
    """训练一个模型，记录逐 epoch 指标并保存验证集最优权重"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total
        val_acc, val_loss = evaluate(model, val_loader)

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"{name}_best.pt"))
        print(f"[{name}] epoch {epoch:2d} | "
              f"train_loss {train_loss_avg:.4f} | val_loss {val_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

    print(f"[{name}] 完成，验证集最优 {best_val_acc:.4f} @ epoch {best_epoch}")
    return history, best_val_acc, best_epoch


# ========================== 4. 可视化 ==========================
def plot_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, h in histories.items():
        epochs = range(1, EPOCHS + 1)
        axes[0].plot(epochs, h["train_loss"], ls="-", label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], ls="--", label=f"{name} val")
        axes[1].plot(epochs, h["train_acc"], ls="-", label=f"{name} train")
        axes[1].plot(epochs, h["val_acc"], ls="--", label=f"{name} val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training / Validation Loss"); axes[0].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Training / Validation Accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "curves.png"), dpi=130)
    plt.close(fig)
    print("已保存训练曲线: figures/curves.png")


@torch.no_grad()
def test_evaluate(model, name, test_loader):
    """测试集评估：准确率 + 混淆矩阵 + 正确/错误样本 + 类间混淆分析"""
    model.eval()
    all_preds, all_labels, all_imgs = [], [], []
    for x, y in test_loader:
        out = model(x.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
        all_imgs.extend(x.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_imgs = np.array(all_imgs)

    acc = (all_preds == all_labels).mean()
    print(f"[{name}] 测试集准确率: {acc:.4f}")

    # 混淆矩阵
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"confusion_{name}.png"), dpi=130)
    plt.close(fig)
    print(f"已保存混淆矩阵: figures/confusion_{name}.png")

    # 每类召回率
    print(f"[{name}] 各类召回率:")
    for c in range(10):
        mask = all_labels == c
        recall = (all_preds[mask] == c).mean()
        print(f"  {CLASSES[c]:<10} {recall:.4f}")

    # 预测正确/错误样本（各 8 张）
    wrong_idx = np.where(all_preds != all_labels)[0]
    correct_idx = np.where(all_preds == all_labels)[0]
    rng = np.random.default_rng(SEED)
    show_sample_grid(all_imgs, all_labels, all_preds,
                     rng.choice(correct_idx, size=min(8, len(correct_idx)), replace=False),
                     f"samples_correct_{name}.png", "Correct Predictions")
    show_sample_grid(all_imgs, all_labels, all_preds,
                     rng.choice(wrong_idx, size=min(8, len(wrong_idx)), replace=False),
                     f"samples_wrong_{name}.png", "Wrong Predictions")

    # cat/dog 混淆分析
    pair = ("cat", "dog")
    ci, di = CLASSES.index(pair[0]), CLASSES.index(pair[1])
    cat_as_dog = ((all_labels == ci) & (all_preds == di)).sum()
    dog_as_cat = ((all_labels == di) & (all_preds == ci)).sum()
    cat_total = (all_labels == ci).sum()
    dog_total = (all_labels == di).sum()
    print(f"[{name}] {pair[0]}->{pair[1]}: {cat_as_dog}/{cat_total} "
          f"({cat_as_dog/cat_total:.2%}); {pair[1]}->{pair[0]}: "
          f"{dog_as_cat}/{dog_total} ({dog_as_cat/dog_total:.2%})")
    return acc, {"cm": cm.tolist(), "acc": float(acc),
                 "cat_as_dog": int(cat_as_dog), "dog_as_cat": int(dog_as_cat),
                 "cat_total": int(cat_total), "dog_total": int(dog_total)}


def show_sample_grid(imgs, labels, preds, idx, fname, title):
    imgs = imgs * np.array(STD).reshape(1, 3, 1, 1) + np.array(MEAN).reshape(1, 3, 1, 1)
    imgs = np.clip(imgs, 0, 1).transpose(0, 2, 3, 1)
    n = len(idx)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        ax.imshow(imgs[i])
        t, p = CLASSES[labels[i]], CLASSES[preds[i]]
        color = "green" if t == p else "red"
        ax.set_title(f"true={t}\npred={p}", fontsize=9, color=color)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"{title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"已保存样本图: figures/{fname}")


# ========================== 主流程 ==========================
def main():
    set_seed(SEED)
    print(f"设备: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    train_full, train_loader, val_loader, test_loader = prepare_data()
    show_data_samples(train_full)

    # ---- 模型结构与参数量 ----
    base_model = BaseCNN()
    improved_model = ImprovedCNN()
    print("\n===== 基础 CNN 各层形状 =====")
    show_layer_shapes(base_model)
    print(f"基础 CNN 参数量: {count_params(base_model):,}")
    print("\n===== 改进 CNN 各层形状 =====")
    show_layer_shapes(improved_model)
    print(f"改进 CNN 参数量: {count_params(improved_model):,}")

    # ---- 训练两个模型（相同数据划分、轮数、批大小、优化器） ----
    histories = {}
    results = {}
    for model, name in ((base_model, "base"), (improved_model, "improved")):
        print(f"\n########## 训练 {name} ##########")
        hist, best_val, best_ep = train_model(model, name, train_loader, val_loader)
        histories[name] = hist
        results[name] = {"best_val_acc": best_val, "best_epoch": best_ep,
                         "params": count_params(model)}
        # 加载验证集最优权重后做测试集评估
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{name}_best.pt"),
                                         map_location=DEVICE))
        acc, eval_info = test_evaluate(model, name, test_loader)
        results[name]["test_acc"] = acc
        results[name]["eval"] = eval_info

    plot_curves(histories)

    # ---- 汇总 ----
    print("\n===== 最终对比 =====")
    for name in ("base", "improved"):
        r = results[name]
        print(f"{name:8s} | 验证集 {r['best_val_acc']:.4f} | "
              f"测试集 {r['test_acc']:.4f} | 参数量 {r['params']:,}")

    with open(os.path.join(BASE_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"history": histories, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("结果已保存: results.json")


if __name__ == "__main__":
    main()
