from dataset.read_txt import read_folder_to_dict
from dataset.encode import encode_text, hash_fontname

import torch

def collate_fn(batch):
    # batch: list of (input, label) tuples
    inputs, labels = zip(*batch)  # #  batch * [sub_seq_len, d_model] 和 batch * [sub_seq_len]

    batch_size = len(inputs)
    seq_lengths = [x.shape[0] for x in inputs]
    max_len = max(seq_lengths)  # 不排序时使用实际最大长度
    d_model = inputs[0].shape[1]

    # 初始化 padded tensors
    padded_inputs = torch.zeros(batch_size, max_len, d_model)
    padded_labels = torch.full((batch_size, max_len), fill_value=-100)  # ignore_index for CrossEntropyLoss
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)  # mask: True 表示有效位置

    for i, (inp, lbl) in enumerate(zip(inputs, labels)):
        seq_len = seq_lengths[i]
        padded_inputs[i, :seq_len, :] = inp
        padded_labels[i, :seq_len] = lbl
        attention_mask[i, :seq_len] = 1  # 有效位置设为 True
    # padded_inputs = padded_inputs.to("cuda")
    # padded_labels = padded_labels.to("cuda")
    # attention_mask = attention_mask.to("cuda")

    return padded_inputs, padded_labels, attention_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, device="cuda", test_len=4, hash_len=1000):
        self.device = device
        self.test_len = test_len
        self.hash_len = hash_len
        self.data = []  # 每个元素是一个 [seq_len, d_model] 的 Tensor
        self.labels = []  # 每个元素: [seq_len]

        txt_dict = read_folder_to_dict(dir)  # {文件名: {属性: [列表]}}

        for _, attr_dict in txt_dict.items():
            texts = attr_dict["text"]
            fontnames = attr_dict["fontname"]
            degrees = attr_dict["degree"]
            other_features = [
                attr_dict["fontsize"], attr_dict["linewidth"],
                attr_dict["left"], attr_dict["center"],
                attr_dict["width"], attr_dict["height"],
                attr_dict["pageid"]
            ]

            feature_list = []
            label_list = []

            for i in range(len(texts)):
                # 编码文本（前4个字符）
                text_encoded = encode_text(texts[i], max_len=test_len)  # [test_len]
                # 哈希字体
                font_encoded = hash_fontname(fontnames[i], hash_len=hash_len)  # [1]
                # 数值特征
                numeric = [float(feat[i]) if feat[i] is not None else 0.0 for feat in other_features]  # [7]

                # 拼成完整特征向量
                full_feature = text_encoded + [font_encoded] + numeric  # [d_model]：text_len + 1 + 7 = 12 (if text_len == 4)
                feature_list.append(full_feature)

                # 添加 degree 标签（转 int）
                try:
                    label_list.append(int(float(degrees[i])))
                except:
                    label_list.append(0)

            tensor = torch.tensor(feature_list, dtype=torch.float32) # [seq_len, d_model]
            if len(tensor.shape) == 1:
                tensor.squeeze(0)
            label_tensor = torch.tensor(label_list, dtype=torch.long) # [seq_len]
            self.data.append(tensor)
            self.labels.append(label_tensor)

        self.length = len(self.data)  # 文件数量
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]   # ([seq_len, d_model], [seq_len])

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device="cuda", max_len=512):
        """
        data: list of [seq_len, d_model]
        labels: list of [seq_len]
        """
        self.samples = []
        for x, y in dataset:
            seq_len = x.shape[0]
            for start in range(0, seq_len, max_len):
                end = min(start + max_len, seq_len)
                self.samples.append((x[start:end], y[start:end]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # ([sub_seq_len, d_model], [sub_seq_len])

    
