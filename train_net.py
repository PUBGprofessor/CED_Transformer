import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.data_process import Dataset
from model.net import Net

def evaluate(model, dataset, device):
    model.eval()
    total = 0
    correct = 0
    correct_body = 0
    total_body = 0
    correct_head = 0
    total_head = 0

    with torch.no_grad():
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # [B, C]
            preds = torch.argmax(outputs, dim=1)

            # 总体准确率
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 正文（label == 0）
            body_mask = labels == 0
            correct_body += (preds[body_mask] == 0).sum().item()
            total_body += body_mask.sum().item()

            # 目录（label >= 1）
            head_mask = labels >= 1
            correct_head += (preds[head_mask] == labels[head_mask]).sum().item()
            total_head += head_mask.sum().item()

    acc = correct / total if total > 0 else 0
    body_acc = correct_body / total_body if total_body > 0 else 0
    head_acc = correct_head / total_head if total_head > 0 else 0

    return acc, body_acc, head_acc

# ========== 设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
learning_rate = 1e-3

# ========== 数据 ==========
dataset = Dataset("./data/train", device)
test_dataset = Dataset("./data/val", device)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== 模型 ==========
net = Net().to(device)

# ========== 损失和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ========== 训练 ==========
for epoch in range(epochs):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataset:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = net(inputs)  # [batch_size, 10] 未softmax

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失与准确率
        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    acc, body_acc, head_acc = evaluate(net, test_dataset, device)
    print(f"Test Accuracy - Total: {acc:.4f}, Body: {body_acc:.4f}, Head: {head_acc:.4f}")
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, 目录: {head_acc:.4f}, 正文: {body_acc:.4f}")
