import os
import torch
from torch.utils.data import DataLoader
from dataset.data_process import Dataset, collate_fn, TransformerDataset
from model.Transformer import TransformerCED

# 模型参数（保持和训练一致）
BATCH_SIZE = 16
MAX_LEN = 512
SRC_VOCAB_SIZE = 50
TRG_VOCAB_SIZE = 10
D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 8
D_FF = 128
DROPOUT = 0.2
EMB_DIM = 16
TEXT_DIM = 4
OTHER_DIM = 8
MODEL_PATH = "./output/model/transformer/v6/epoch50.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = TransformerCED(
    src_vocab=SRC_VOCAB_SIZE, trg_vocab=TRG_VOCAB_SIZE,
    d_model=D_MODEL, N=NUM_LAYERS, heads=NUM_HEADS,
    d_ff=D_FF, dropout=DROPOUT,
    emb_dim=EMB_DIM, text_dim=TEXT_DIM, other_feat_dim=OTHER_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

# 输出格式为：内容**预测标签
def predict_and_add_labels(folder_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    for fname in files:
        input_file = os.path.join(folder_path, fname)
        output_file = os.path.join(output_path, fname)

        print(f"Processing {fname}...")

        raw_dataset = Dataset(input_file, device=device)
        dataset = TransformerDataset(raw_dataset, max_len=MAX_LEN)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        all_lines = []
        # 读取原始文本（每行是逗号分隔的字段）
        with open(input_file, 'r', encoding='utf-8') as fin:
            original_lines = [line.strip() for line in fin if line.strip()]

        idx = 0
        with torch.no_grad():
            for inputs, labels, mask in loader:
                inputs = inputs.to(device)
                mask = mask.to(device)

                outputs = model(inputs, mask)  # [1, L, C]
                preds = outputs.argmax(dim=-1)[0].tolist()

                for i in range(len(preds)):
                    label = preds[i]
                    if label == 0:
                        idx += 1
                        continue  # 跳过正文行
                    if idx < len(original_lines):
                        fields = original_lines[idx].split(",")
                        text = fields[-1].strip()
                        all_lines.append(f"{text} **{label}")
                    idx += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_lines) + '\n')

# === 配置路径 ===
input_folder = "./data/val_out"      # 替换为你的输入路径
output_folder = "./output/data/labeled_txt"  # 输出路径

if __name__ == "__main__":
    predict_and_add_labels(input_folder, output_folder)
