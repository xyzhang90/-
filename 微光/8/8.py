# -*- coding: utf-8 -*-
"""
ml-8  MiniGPT：从零手写 Transformer（Decoder-Only 语言模型）
数据集：tinyshakespeare（字符级）
参考骨架：课程题目 glimmer.org.cn/ml/8/
"""

import os
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# ======================== 0. 数据集获取（若 input.txt 不存在则自动下载） ========================
DATA_URLS = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://codeload.github.com/karpathy/char-rnn/zip/refs/heads/master",  # zip，需解压
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
]


def ensure_dataset(path="input.txt"):
    """如果本地没有 input.txt，尝试从多个镜像下载（保持脚本开箱即用）。"""
    if os.path.exists(path):
        return
    import zipfile
    import io
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if url.endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(data))
                name = [n for n in z.namelist() if n.endswith("input.txt")][0]
                data = z.read(name)
            if len(data) > 100_000:
                with open(path, "wb") as f:
                    f.write(data)
                print(f"[INFO] 数据集已自动下载到 {path}")
                return
        except Exception as e:
            print(f"[WARN] 从 {url} 下载失败: {e}")
    raise FileNotFoundError("未找到 input.txt 且自动下载失败，请手动下载数据集后重试")


ensure_dataset("input.txt")

# ======================== 1. 超参设置（与题目一致） ========================
batch_size = 16     # 批处理大小
block_size = 32     # 上下文长度
max_iters = 5000    # 训练轮数
learning_rate = 1e-3  # 学习率
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 64         # embedding维度
n_head = 4          # 多头注意力头数
n_layer = 4         # transformer层数
dropout = 0.0       # dropout率
torch.manual_seed(42)  # 随机种子，保证可复现

# 以下两个参数用于周期性评估，保证估计损失平滑
eval_interval = 500  # 每隔多少步评估一次
eval_iters = 200     # 每次评估用多少批求平均

print(f"运行设备: {device}")

# ======================== 2. 数据预处理 ========================
# 读取莎士比亚文本
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"数据集总字符数: {len(text):,}")
print(f"前300个字符预览:\n{text[:300]}")

# 构建词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词汇表: {''.join(chars)}")
print(f"词汇表大小: {vocab_size}")

# 简单tokenizer实现
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# 测试编解码
print(f"编码测试: {encode('Hello World!')}")
print(f"解码测试: {decode(encode('Hello World!'))}")

# 训练/验证集切分：前90%训练，后10%验证
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ======================== 3. 数据分批处理 ========================
def get_batch(split):
    """从训练/验证集中随机采样一个批次。

    输入序列 x 与目标序列 y 一一错位（预测下一个字符）。
    """
    data = train_data if split == "train" else val_data
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 每个样本取 [i, i+block_size) 作为输入，[i+1, i+block_size+1) 作为目标
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ======================== 4. 模型架构实现 ========================
class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, head_size):
        super().__init__()
        # QKV投影矩阵
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 下三角mask矩阵（防止看到未来）
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # 1. 计算 Q, K, V
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        # 2. 计算注意力分数（Q @ K^T 并缩放，防止内积过大导致softmax梯度消失）
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        # 3. 应用下三角 mask：屏蔽未来位置
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # 归一化得到注意力权重
        wei = self.dropout(wei)
        # 4. 加权求和
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力：并行处理多个注意力头"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # 输出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头并行计算，然后沿特征维拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈神经网络：两层线性变换 + ReLU 激活，隐层放大4倍"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block：communication（注意力）+ computation（前馈）"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # Pre-LN：残差之前先做 LayerNorm，训练更稳定
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 自注意力子层 + 残差
        x = x + self.sa(self.ln1(x))
        # 前馈子层 + 残差
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """完整的语言模型（Decoder-Only Transformer / MiniGPT）"""

    def __init__(self):
        super().__init__()
        # 每个token对应一个可学习向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 每个位置对应一个可学习向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 堆叠transformer块
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # 最后的LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 映射回词表

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. token embedding + position embedding
        tok_emb = self.token_embedding_table(idx)              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))                    # (T, n_embd)
        x = tok_emb + pos_emb                                  # (B, T, n_embd)

        # 2. 通过transformer块
        x = self.blocks(x)                                     # (B, T, n_embd)
        x = self.ln_f(x)                                       # (B, T, n_embd)

        # 3. 输出预测
        logits = self.lm_head(x)                               # (B, T, vocab_size)

        # 4. 计算损失（如果有target）
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """文本生成：给定上下文，逐个采样新token"""
        for _ in range(max_new_tokens):
            # 裁剪到block_size（上下文不能超过模型见过的最大长度）
            idx_cond = idx[:, -block_size:]
            # 预测下一个token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只要最后一个时间步
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token（带随机性，而非贪心）
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ======================== 5. 模型评估 ========================
@torch.no_grad()
def estimate_loss():
    """评估训练和验证损失（取多个批次的平均，减少波动）"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ======================== 6. 模型训练 ========================
if __name__ == "__main__":
    # 实例化模型
    model = BigramLanguageModel().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
          f" ({sum(p.numel() for p in model.parameters()):,} 参数)")

    # 训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []

    for iter in range(max_iters):
        # 1. 获取batch
        xb, yb = get_batch("train")

        # 2-3. 前向传播并计算损失
        logits, loss = model(xb, yb)

        # 4-5. 反向传播并更新参数
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 6. 定期评估
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            print(f"step {iter:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

    # ======================== 7. 绘制损失曲线 ========================
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="#FF6B6B")
    plt.plot(val_losses, label="Validation Loss", color="#4ECDC4")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Progress: Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(train_losses)), [i * eval_interval for i in range(len(train_losses))])
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
    # 在 Jupyter 中如需显示曲线，可取消下一行注释
    # plt.show()

    # ======================== 8. 生成莎士比亚风格文本 ========================
    print("AI莎士比亚开始创作...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
    print(generated_text)

    # ======================== 9. 结果分析 ========================
    print("\n================= 结果分析 =================")
    print(f"Validation Loss 收敛值: {val_losses[-1]:.4f}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"MiniGPT 参数量: {param_count:,} ({param_count/1e6:.3f}M)  <1M: {param_count < 1_000_000}")
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        print(f"显存占用: {mem_used:.3f} GB  <1G: {mem_used < 1.0}")
    else:
        print("当前运行在 CPU 上，无显存占用（若在 GPU 上运行可查看 torch.cuda.max_memory_allocated）")
    print("==========================================")
