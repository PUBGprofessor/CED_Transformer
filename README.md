## 环境：

- **pytorch**

## 文件结构：

```
CED_transformer/
├── data/
│   ├── failed/
│   ├── train/
│   ├── val/
│   └── val_out/
│
├── dataset/
│   └── ... 
│
├── model/
│   └── ...
│
├── output/
│   ├── data/
│   │   └── labeled_txt/
│   └── model/
│       ├── net/
│       │   └── v1/
│       ├── transformer/
│       |    └── v1/
|       └── transformer/
│           └── v1/
│
├── utils/
│   └── ...
│
├── test.py
├── pre_test.py
├── train_net.py
├── train_Transformer.py
├── train_preTransformer.py
└── README.md
```

## RUN：

### 1.数据

将CED获得的标注完成后的**特征txt**：

**"degree_right, fontsize, fontname,linewidth,left(左端),center(中点),width(字块所占宽度),height(字块所占高度), pageid, 文本"**

...

分放在**.\data\train**和**.\data\val**下作为训练集和测试集

**已标注数据下载**：http://www.rmclass.cn/data.zip

### 2.训练

**（1）train_net.py ：不建议**

​	训练神经网络（稳定后准确率约70%）

**（2）train_Transformer.py：** 

​	训练Transformer（稳定后准确率约80%）

​	目录Acc: 目录识别正确率， 正文Acc：正文识别正确率

   **(3)** **train_preTransformer.py + train_Transformer.py**:

​	设置的Transformer输入的seq长度最大为512，但一般的财报文件的句子组的长度为几千到上万，先使用preTransformer过滤掉较大可能性的正文句子，然后有transformer进行分类

### 3. 测试

```
train test.py
```

须在py中改参数：

```
input_folder = "./data/val_out"    # 替换为你的输入路径,val_out中包含的txt为所要提取目录的pdf文件提取出的特征txt，degree(第一列)全设0即可

output_folder = "./output/data/labeled_txt"  # 输出路径，提取完成的目录txt
```



## 后续优化：

1. 设置的Transformer输入的seq长度最大为512，但一般的财报文件的句子组的长度为几千到上万，计划先使用普通神经网络过滤掉较大可能性的正文句子，然后有transformer进行分类（已实现）
2. 提取更多句子特征，或使用全部句子文本（目前只用每个句子的前4个字符：可能包含关键目录特征如序号等），增大输入维度