# -*- coding: utf-8 -*-
"""
ml-7 循环神经网络实战项目：IMDB 电影评论情感分析（BiLSTM 二分类）

实现内容：
  1. DataProcessor      文本清洗 / 词汇表构建 / 转索引 / 填充
  2. SentimentDataset   把 pad 后的等长数组包成 torch Dataset
  3. create_data_loaders  80/10/10 划分，返回 (train/val/test_loader, processor)
  4. BiLSTMClassifier   双向 LSTM 分类模型
  5. SentimentTrainer   训练 / 评估 / 早停 / 学习率调度 / 保存最佳模型
  6. main()             从 CSV 到训练、测试、评估、绘图、错误分析

运行方式：
  python ml-7-姓名-学号.py [数据csv路径] [epochs] [--no-plot|--save-plots]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows 下避免 OMP 库冲突

import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# 无显示环境时：不弹窗，改为把图保存成 PNG 文件
SAVE_PLOTS = '--save-plots' in sys.argv
if SAVE_PLOTS:
    import matplotlib
    matplotlib.use('Agg')   

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================== 全局配置 ===========================
DATA_PATH = "IMDB Dataset.csv"   # 数据文件路径（可从命令行第一个参数覆盖）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CELL: 1-工具函数（题目已提供，直接使用）=====

def calculate_metrics(y_true, y_pred, num_classes):
    """计算所有评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    average = 'binary' if num_classes == 2 else 'macro'
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return cm


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.legend()
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.legend()
    plt.tight_layout(); plt.show()


# ===== CELL: 2-DataProcessor（数据预处理核心）=====

class DataProcessor:
    """
    文本 -> 清洗 -> 词汇表 -> 索引 -> 填充 的完整处理流程。
    word2idx / idx2word 存在 self 里，避免函数之间到处传参。
    fit 只在训练集上调用一次（防止数据泄漏），transform 用于把任意文本转成等长数组。
    """

    def __init__(self, max_vocab_size=10000, max_seq_length=128):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        # 预留两个特殊 token：<PAD> 用于填充，<UNK> 用于词汇表外的词
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def clean_text(self, text):
        """
        清洗文本：先去 HTML 标签 -> 转小写 -> 去标点 -> 规范化空格。
        注意操作顺序：必须先去掉 <br /> 再去标点，否则 <br /> 里的字母会残留成 "br"。
        输入: "This movie is <br /> great!! I love it."
        期望输出: "this movie is great i love it"
        """
        text = re.sub(r'<br\s*/?>', ' ', text)   # 1. 把 <br /> 换成空格
        text = text.lower()                       # 2. 转小写
        text = re.sub(r'[^a-z\s]', ' ', text)     # 3. 非字母转空格（去标点）
        text = ' '.join(text.split())             # 4. 规范化多个空格为一个
        return text

    def build_vocab(self, texts):
        """
        在训练集上构建词汇表（写入 self.word2idx / self.idx2word）。
        - <PAD>=0, <UNK>=1 已在 __init__ 中预留，不要重复加
        - 用 Counter 统计词频，取最常见的 max_vocab_size - 2 个词
        - 实际词的索引从 2 开始
        """
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        # 取出现频率最高的 max_vocab_size - 2 个词
        for word, _ in counter.most_common(self.max_vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequence(self, text):
        """
        文本 -> 索引序列（用 self.word2idx），词汇表外的词用 <UNK>(=1) 代替。
        输入: "this movie is great"
        输出: [2, 3, 4, 5]（假设这些词都在词汇表中）
        """
        return [self.word2idx.get(w, 1) for w in text.split()]

    def pad_sequences(self, sequences):
        """
        序列填充/截断到 self.max_seq_length（前面补 0，末尾保留）。
        策略：
        - 长度 < max_len：前面补 0，有效内容靠右
        - 长度 > max_len：截断保留末尾（开头被截掉）
        示例 (max_len=4)：[[2,3], [4,5,6,7,8]] -> [[0,0,2,3], [5,6,7,8]]
        返回 numpy.ndarray，形状 [n, max_len]，dtype=int64
        （torch.LongTensor 要求规则形状，不能直接转 list of 变长 lists）
        """
        padded = np.zeros((len(sequences), self.max_seq_length), dtype=np.int64)
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            if len(seq) >= self.max_seq_length:
                padded[i] = seq[-self.max_seq_length:]   # 保留末尾
            else:
                padded[i, -len(seq):] = seq             # 前面补 0
        return padded

    def fit(self, train_texts):
        """在训练集上构建词汇表（只在训练集调用一次）"""
        cleaned = [self.clean_text(t) for t in train_texts]
        self.build_vocab(cleaned)
        return self

    def transform(self, texts):
        """把文本列表转为 pad 后的 numpy array（给 SentimentDataset 用）"""
        cleaned = [self.clean_text(t) for t in texts]
        sequences = [self.text_to_sequence(t) for t in cleaned]
        return self.pad_sequences(sequences)


# ===== CELL: 3-SentimentDataset 与 create_data_loaders=====

class SentimentDataset(Dataset):
    """
    关键：传入的 texts 必须已经是 pad 后的等长 array
    （torch.LongTensor 要求规则形状，不能是 list of 变长 lists）。
    """
    def __init__(self, texts, labels):
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def create_data_loaders(texts, labels, batch_size=64, random_state=42):
    """
    划分 train/val/test = 80/10/10，创建 DataLoader。
    返回: (train_loader, val_loader, test_loader, processor)
      processor 里有 word2idx，模型需要用 len(processor.word2idx) 作为 vocab_size
    步骤:
      1. train_test_split 切出 10% test
      2. 再在剩下的里切出 1/9 作为 val（1/9 * 0.9 = 0.1）
      3. DataProcessor.fit(X_train) —— 只在训练集上建词表
      4. processor.transform 转换三份数据
      5. 包成 SentimentDataset + DataLoader（train shuffle=True，val/test shuffle=False）
    """
    # 1. 先切出 10% 的测试集（按类别分层，保证正负样本比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.1, random_state=random_state, stratify=labels)

    # 2. 在剩下的 90% 里再切出 1/9 作为验证集（1/9 * 0.9 = 0.1）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=1 / 9, random_state=random_state, stratify=y_train)

    # 3. 只在训练集上构建词汇表
    processor = DataProcessor()
    processor.fit(X_train)

    # 4. 转换三份数据
    X_train_pad = processor.transform(X_train)
    X_val_pad = processor.transform(X_val)
    X_test_pad = processor.transform(X_test)

    # 5. 包成 Dataset + DataLoader
    train_loader = DataLoader(SentimentDataset(X_train_pad, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SentimentDataset(X_val_pad, y_val),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SentimentDataset(X_test_pad, y_test),
                             batch_size=batch_size, shuffle=False)
    # 把测试集的原始文本挂在 processor 上，方便之后做错误样本分析
    processor.test_texts = X_test
    return train_loader, val_loader, test_loader, processor


# ===== CELL: 4-BiLSTMClassifier（模型）=====

class BiLSTMClassifier(nn.Module):
    """双向 LSTM 情感分类模型：Embedding -> BiLSTM -> Dropout -> Linear"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes=2, dropout=0.3, pad_idx=0, bidirectional=True):
        super().__init__()
        # 1. Embedding 层：把词索引转为向量，[batch, seq_len] -> [batch, seq_len, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 2. 双向 LSTM：batch_first=True 输入形状 [batch, seq_len, embed_dim]
        #    dropout 只对 num_layers > 1 时层与层之间生效（单层传 0 避免警告）
        #    bidirectional 可切换（默认双向；设 False 即单向 LSTM，用于 3.2 对比实验）
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)

        # 3. Dropout 层（在分类头之前随机失活，防止过拟合）
        self.dropout = nn.Dropout(dropout)

        # 4. 全连接层：双向时输入维度是 hidden_dim * 2（前向+后向拼接），单向为 hidden_dim
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        前向传播，维度追踪：
            x:        [batch, seq_len]
            embedded: [batch, seq_len, embed_dim]
            lstm_out: [batch, seq_len, hidden_dim * 2]（双向；单向为 hidden_dim）
            pooled:   [batch, hidden_dim * 2]（双向；单向为 hidden_dim，取最后一个时间步）
            output:   [batch, num_classes]
        """
        embedded = self.embedding(x)                       # [b, s] -> [b, s, e]
        lstm_out, (h_n, c_n) = self.lstm(embedded)         # [b, s, 2h]
        # 取最后一个时间步的输出作为序列表示
        # 思考：前面补 0 后有效内容靠右，最后一个时间步的隐状态最能代表整个序列
        pooled = lstm_out[:, -1, :]                        # [b, 2h]
        pooled = self.dropout(pooled)
        output = self.fc(pooled)                           # [b, num_classes]
        return output


# ===== CELL: 5-SentimentTrainer（训练器）=====

class SentimentTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-5,
                 device=DEVICE, grad_clip=1.0):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        # 分类任务用交叉熵（CrossEntropyLoss 内部自带 softmax + log）
        # 为什么不用 MSE？分类输出是类别概率分布，交叉熵对概率分布间的差异
        # 更敏感、梯度更稳；MSE 配合 softmax 会出现梯度饱和，收敛慢。
        self.criterion = nn.CrossEntropyLoss()
        # weight_decay 是 L2 正则，抑制过拟合
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # 验证 loss 不再下降时把学习率减半（选做：学习率调度）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2)

    def train_epoch(self, train_loader):
        """训练一个 epoch，返回 (平均loss, 准确率)"""
        self.model.train()                      # 开启 dropout
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()          # 关键：每个 batch 前清零梯度
            output = self.model(x)              # [batch, num_classes]
            loss = self.criterion(output, y)
            loss.backward()
            # 梯度裁剪：防止梯度爆炸
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total += len(y)
        return total_loss / total, total_correct / total

    @torch.no_grad()
    def evaluate(self, data_loader):
        """评估，返回 (平均loss, 预测列表, 真实标签列表)"""
        self.model.eval()                       # 关闭 dropout
        total_loss, total = 0.0, 0
        all_preds, all_labels = [], []
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            total_loss += loss.item() * len(y)
            total += len(y)
            all_preds.extend(output.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
        return total_loss / total, all_preds, all_labels

    def train(self, train_loader, val_loader, num_epochs=10, patience=3,
              best_model_path="best_model.pt"):
        """
        完整训练流程，返回训练历史 dict。
        含：每轮训练+验证、记录曲线、早停（验证 loss 连续 patience 轮不降即停）、
            学习率调度、保存验证集上最优模型。
        """
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        bad_epochs = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_preds, val_labels = self.evaluate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            # 学习率调度：验证 loss 不再下降时衰减
            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch:2d}/{num_epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"LR {self.optimizer.param_groups[0]['lr']:.2e}")

            # 保存验证集上最优的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                bad_epochs += 1
                # 早停：连续 patience 轮验证 loss 不下降就停
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}（验证 loss 连续 {patience} 轮未下降）")
                    break

        # 恢复到验证集上最优的权重
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print(f"已恢复验证集最优模型（val_loss={best_val_loss:.4f}），保存在 {best_model_path}")
        return history


# ===== CELL: 6-主流程（main）=====

def decode_sequence(seq, processor):
    """把索引序列解码回单词（用于错误样本分析展示）"""
    return ' '.join(processor.idx2word.get(int(i), '<UNK>') for i in seq)


def main():
    # 1. 命令行参数：数据路径 / 训练轮数 / --no-plot / --save-plots
    data_path, num_epochs, do_plot = DATA_PATH, 10, True
    for a in sys.argv[1:]:
        if a == '--no-plot':
            do_plot = False
        elif a == '--save-plots':
            pass  # 已在文件头处理（SAVE_PLOTS）
        elif a.isdigit():
            num_epochs = int(a)
        else:
            data_path = a

    # 2. 加载数据：review + sentiment 两列，sentiment -> 0 (negative) / 1 (positive)
    if not os.path.exists(data_path):
        print(f"找不到数据文件：{data_path}")
        print("请先从 QQ 群下载 IMDB Dataset 文件夹，把 IMDB Dataset.csv 放到本脚本同目录后重跑。")
        return
    try:
        df = pd.read_csv(data_path)
    except Exception:
        # 某些版本的数据集引号不完整，用 python 引擎更宽容
        df = pd.read_csv(data_path, engine='python')
    print(f"原始数据量：{len(df)} 条")

    # 数据探索：类别分布 / 重复项 / 文本长度分布
    print(df['sentiment'].value_counts().to_string())
    # 去掉完全重复的评论（Kaggle 原数据有重复，去重有助于精度提升）
    df = df.drop_duplicates(subset='review')
    print(f"去除重复评论后：{len(df)} 条")

    texts = df['review'].astype(str).tolist()
    labels = (df['sentiment'] == 'positive').astype(int).tolist()  # negative=0, positive=1

    # 3. 数据预处理 + 划分数据集
    train_loader, val_loader, test_loader, processor = create_data_loaders(texts, labels)
    print(f"词汇表大小：{len(processor.word2idx)}，序列最大长度：{processor.max_seq_length}")

    # 4. 创建模型（vocab_size 从 processor 拿）
    model = BiLSTMClassifier(
        vocab_size=len(processor.word2idx),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        pad_idx=processor.word2idx['<PAD>'],
    )
    print(f"设备：{DEVICE}；模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练
    trainer = SentimentTrainer(model, lr=1e-3, weight_decay=1e-5, device=DEVICE)
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # 6. 测试集评估
    test_loss, test_preds, test_labels = trainer.evaluate(test_loader)
    metrics = calculate_metrics(test_labels, test_preds, num_classes=2)
    print("\n========== 测试集结果 ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 7. 训练曲线 + 混淆矩阵（--save-plots 时保存 PNG，否则弹窗显示）
    if do_plot:
        plot_training_history(history['train_loss'], history['val_loss'],
                              history['train_acc'], history['val_acc'])
        if SAVE_PLOTS:
            plt.gcf().savefig('训练曲线.png', dpi=120, bbox_inches='tight')
            plt.close('all')
        plot_confusion_matrix(test_labels, test_preds, class_names=['negative', 'positive'])
        if SAVE_PLOTS:
            plt.gcf().savefig('混淆矩阵.png', dpi=120, bbox_inches='tight')
            plt.close('all')
            print("已保存图片：训练曲线.png / 混淆矩阵.png")

    # 8. 错误样本分析（喂给报告 Q1）
    print("\n========== 错误样本示例（用于报告错误分析） ==========")
    test_texts_raw = list(processor.test_texts)  # 测试集原始文本（create_data_loaders 时挂载）
    wrong_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_labels)) if p != t]
    print(f"测试集共 {len(test_labels)} 条，其中预测错误 {len(wrong_idx)} 条")
    for i in wrong_idx[:5]:
        snippet = test_texts_raw[i][:120].replace('\n', ' ')
        print(f"\n[错例] 真实={'positive' if test_labels[i] else 'negative'} | "
              f"预测={'positive' if test_preds[i] else 'negative'}")
        print(f"  评论: {snippet}...")
    print("\n提示：想看清每类错误比例，可用 sklearn 的 classification_report。")


if __name__ == '__main__':
    main()
