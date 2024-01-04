bert_pro 
--------------------------------------------------------------
此文件夹中的代码用于生成评论单独的bert编码 原文对应的bertr编码以及原文和评论组合起来的bert编码

作用：用于生成文本的编码以及使用LSTM生成一版预测的结果用于与其他模型ensemble

各个文件夹文件介绍：
--------------------------------------------------------------

\bert_uncased 用于存放预训练好的bert模型

\Ma-Weibo_init:
\data:测试集添加了label列，并设置为了全为rumor或者nonrumor，便于使用train的代码生成需要的文本编码
ours:用于存放预处理好将要进行bert编码的数据
embedding_new:生成出来的编码好的pkl，用于后续的处理，也存放了整个模型的预测结果，用于后续与其他的模型一同ensemble
process_repost_comments_basic.py:简单的文本预处理

\trained_bert:存放使用训练集微调好的bert
trained_bert_comments：使用comment数据微调的bert
trained_bert_text_comments：使用comment和原始推文的数据微调的bert

\trained_models:存放训练好的模型（在微调bert的基础上再进行训练的模型）


主要代码：
--------------------------------------------------------------

train_bert.py：用于微调bert并保存

train_gain_embedding.py：用于生成单独的文本编码，用于后续处理

train_embedding.py：用于生成模型最终二值化之前的结果并保存，用于与其他的模型一同ensemble

utils.py：定义了一些函数，用于划分编码的句子的长度









