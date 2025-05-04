import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import time

from dataset.data_process import Dataset, collate_fn, TransformerDataset
from model.Transformer import TransformerCED

# 0. 超参设置
BATCH_SIZE = 16
MAX_LEN = 512
SRC_VOCAB_SIZE = 50
TRG_VOCAB_SIZE = 10
D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 8
D_FF = 128
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 50
WEIGHT_0 = 0.2
EMB_DIM = 16
TEXT_DIM = 4
OTHER_DIM = 8
ckpt_path = r"./output/model/transformer/v6"

os.makedirs(ckpt_path, exist_ok=True)

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_count = 0

    correct_body = 0  # label == 0
    total_body = 0

    correct_head = 0  # label >= 1
    total_head = 0

    with torch.no_grad():
        for inputs, labels, mask in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(inputs, mask)  # [B, L, C]
            preds = outputs.argmax(dim=-1)  # [B, L]

            preds = preds.view(-1)
            labels = labels.view(-1)

            valid = labels != -100
            valid_preds = preds[valid]
            valid_labels = labels[valid]

            total_correct += (valid_preds == valid_labels).sum().item()
            total_count += valid_labels.size(0)

            # 正文（label == 0）
            body_mask = valid_labels == 0
            correct_body += (valid_preds[body_mask] == 0).sum().item()
            total_body += body_mask.sum().item()

            # 目录（label >= 1）
            head_mask = valid_labels >= 1
            correct_head += (valid_preds[head_mask] == valid_labels[head_mask]).sum().item()
            total_head += head_mask.sum().item()

    overall_acc = total_correct / total_count if total_count > 0 else 0
    body_acc = correct_body / total_body if total_body > 0 else 0
    head_acc = correct_head / total_head if total_head > 0 else 0

    return overall_acc, body_acc, head_acc

# 1. 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time() 

# 2. 加载数据
raw_dataset = Dataset("./data/train", device=device)
processed_dataset = TransformerDataset(raw_dataset, max_len=MAX_LEN)
dataloader = DataLoader(processed_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# 6. 加载测试集
test_raw_dataset = Dataset("./data/val", device=device)
test_dataset = TransformerDataset(test_raw_dataset, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

data_time = time.time()
print(f"数据集加载完成, 耗时{data_time - start_time}s")

# 3. 初始化模型
model = TransformerCED(src_vocab=SRC_VOCAB_SIZE, trg_vocab=TRG_VOCAB_SIZE, d_model=D_MODEL, N=NUM_LAYERS, heads=NUM_HEADS, d_ff=D_FF, dropout=DROPOUT
                        , emb_dim=EMB_DIM, text_dim=TEXT_DIM, other_feat_dim=OTHER_DIM)
model = model.to(device)

# 4. 损失函数 & 优化器
def get_class_weights(num_classes=10, device='cuda'):
    # 类别 0 的权重设置为最小，比如 0.2
    weight_0 = WEIGHT_0
    # 其余类别的权重从 1.0 递减到 0.2（共 num_classes-1 个）
    other_weights = torch.linspace(1.0, 0.2, steps=num_classes - 1)
    # 拼接最终权重向量：[0类权重] + [1~9类权重]
    weights = torch.cat([torch.tensor([weight_0]), other_weights])
    return weights.to(device)

weights = get_class_weights(TRG_VOCAB_SIZE, device)
criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. 训练循环
epochs = EPOCHS
for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    total_samples = 0

    for inputs, labels, mask in dataloader:
        # inputs: [B, L, d_model]
        # labels: [B, L]
        # mask:   [B, L]

        inputs = inputs.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, mask)  # [B, L, num_classes] 未softmax

        # reshape for loss: (B * L, num_classes), (B * L)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)  # inputs.size(0) 是 batch size
        total_samples += inputs.size(0)

    acc = evaluate(model, test_loader, device)
    avg_loss = total_loss / total_samples

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc[0]:.4f}, 目录Acc: {acc[2]:.4f}, 正文Acc: {acc[1]:.4f}, Time: {epoch_time:.2f}s")
    if (epoch + 1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(ckpt_path, f"epoch{epoch+1}.pth"))

end_time = time.time()
total_time = end_time - start_time
print(f"训练完成，总耗时：{total_time:.2f} 秒")
