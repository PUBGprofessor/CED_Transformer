
# 允许的字符集
char_set = list("一二三四五六七八九十1234567890（）()、. ")

# 建立字符到整数的映射，其他字符统一映射为 0
char2id = {c: i+1 for i, c in enumerate(char_set)}
char2id['<other>'] = 0  # 未包含的字符

def encode_text(text, max_len=4):
    # 输入字符串, 返回整数列表
    text = text.strip()[:max_len]  # 最多4个字符
    encoded = []
    for ch in text:
        encoded.append(char2id.get(ch, char2id['<other>']))
    # 补齐长度为 max_len
    while len(encoded) < max_len:
        encoded.append(0)
    return encoded

def hash_fontname(fontname, hash_len=1000):
    # 取 + 后的字体名
    if '+' in fontname:
        fontname = fontname.split('+', 1)[1]
    # 统一映射为一个固定范围的整数（如 0~999）
    return hash(fontname) % hash_len