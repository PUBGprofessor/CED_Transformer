import os

def process_file(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts and len(parts) > 1:
                parts[0] = '0'
                lines.append(','.join(parts))
            else:
                lines.append(line.strip())  # 原样保留空行或异常格式行

    # 保存（可选：备份原文件）
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)
            print(f"Processed: {filename}")

# ==== 用法 ====
if __name__ == '__main__':
    target_folder = '/home/user/mydisk/3DGS_code/CED_Transformer/data/val_out'  # 替换为你的文件夹路径
    process_folder(target_folder)


