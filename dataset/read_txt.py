import os

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def safe_int(x):
    try:
        return int(x)
    except:
        return 0
    
def parse_txt_file(filepath):
    data = {
        "degree": [],
        "fontsize": [],
        "fontname": [],
        "linewidth": [],
        "left": [],
        "center": [],
        "width": [],
        "height": [],
        "pageid": [],
        "text": []
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue  # 跳过格式不完整的行

            try:
                # 前9项是数值 + 字符串，最后是text（注意 text 可能包含逗号）
                degree = safe_int(parts[0])
                fontsize = safe_float(parts[1])
                fontname = parts[2] if parts[2].strip() != 'None' else ''
                linewidth = safe_float(parts[3])
                left = safe_float(parts[4])
                center = safe_float(parts[5])
                width = safe_float(parts[6])
                height = safe_float(parts[7])
                pageid = safe_int(parts[8])
                text = ','.join(parts[9:]).strip()
                # text = ','.join(parts[9:]).strip()[:4]

                data["degree"].append(degree)
                data["fontsize"].append(fontsize)
                data["fontname"].append(fontname)
                data["linewidth"].append(linewidth)
                data["left"].append(left)
                data["center"].append(center)
                data["width"].append(width)
                data["height"].append(height)
                data["pageid"].append(pageid)
                data["text"].append(text)
            except Exception as e:
                print(f"解析出错：{filepath} 中某一行：{e}")
    
    return data

# def read_folder_to_dict(folder_path):
#     all_data = {}

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             full_path = os.path.join(folder_path, filename)
#             all_data[filename[:-4]] = parse_txt_file(full_path)
    
#     return all_data

def read_folder_to_dict(path):
    all_data = {}

    def process_file(file_path):
        if file_path.endswith(".txt"):
            name = os.path.basename(file_path)[:-4]
            all_data[name] = parse_txt_file(file_path)

    if os.path.isfile(path):
        process_file(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            process_file(full_path)

    return all_data
