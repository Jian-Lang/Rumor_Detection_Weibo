## quick_check

运行ensemble_and_output_results/ensemble_and_output.py文件，即可大致复现我们的结果

```bash
python ensemble_and_output_results/ensemble_and_output.py
```

------



## 模型(方法)简单介绍

### 主模型：基于双向情感度分析的微博谣言检测

#### 主要思想：

基于双向情感度，通过情感度分析结果，建模评论 与 原文的情感度之间的关系

于此同时，加上对数据分析得到的各类特征工程进行加强

#### 贡献：

主模型产生了十余个集成的结果

### 辅助模型1：

辅助模型1的介绍在一个单独的文件当中：RumorDetection\bert_pro_readme.md

#### 贡献：

辅助模型1产生8个结果

### 辅助模型2：

互联网公开的一个预训练飞浆模型，通过我们的数据训练后，作为辅助(https://www.paddlepaddle.org.cn/)

#### 贡献：

产生1个预测结果

### 特征工程

#### 基础工程

对各类数据进行预处理，包括特征映射、数据清洗、0 1化、简单的四则运算等等

#### 文本级别

##### 文本编码

通过hugging face上的模型以及辅助模型2，进行文本(评论以及二者结合)的编码 (https://huggingface.co/bert-base-chinese) (https://huggingface.co/owen198/weibo-wmmbert-6)

##### 情感度

通过hugging face上的模型进行情感度的编码 (https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)

##### 句级相似度

通过hugging face上的模型进行相似度的编码 (https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Similarity) (https://huggingface.co/abbasgolestani/ag-nli-DeTS-sentence-similarity-v1)

##### 词级相似度：词袋

通过大量对评论的分析(尤其是训练数据集的谣言评论)，构建词袋，作为一项特征

#### 转发级别

##### 转发链

通过递归算法获得转发链

##### 转发深度

通过转发链，获取转发深度等特征

##### 转发网络

通过转发链，构建转发网络

### 集成学习

最终结果使用了集成学习的思想，集成了主模型、辅助模型的结果。

主模型的集成文件夹当中提供了预训练好的模型 以及 模型的代码，**辅助模型1的相关文件在"RumorDetection\bert_pro"内**，辅助模型2来自飞浆的官方网站开源。

### 注

详细的方法和模型的讲解，将放在决赛的PPT当中，并在决赛当中予以讲解

### 参考文献资料

1. [【NLP实战系列】基于TextCNN/RNN/LSTM微博谣言检测](https://zhuanlan.zhihu.com/p/617675937?utm_campaign=&utm_medium=social&utm_oi=1001917154937176064&utm_psn=1710746055602438144&utm_source=com.ss.android.lark)
2. [谣言检测论文分享](https://blog.csdn.net/weixin_41964296/article/details/129785082?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"129785082"%2C"source"%3A"unlogin"})
3. [谣言检测论文分享A Survey on Fake News and Rumour Detection Techniques](https://blog.csdn.net/weixin_41964296/article/details/131057801?app_version=6.2.3&code=app_1562916241&csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"131057801"%2C"source"%3A"weixin_44956321"}&uLinkId=usr1mkqgl919blen&utm_source=app)
4. [基于评论异常度的新浪微博谣言识别方法](http://www.aas.net.cn/fileZDHXB/journal/article/zdhxb/2020/8/PDF/zdhxb-46-8-1689.pdf)
5. [一种基于微博类型的集成微博谣言识别方法](https://patentimages.storage.googleapis.com/e0/b5/75/fbf802394a27be/CN106202211A.pdf)
6. [在线社会网络谣言检测综述](http://cjc.ict.ac.cn/online/bfpub/cyf-2017126164816.pdf)
7.  **[Implementation-of-VGCN-BERT-for-Rumor-Detection](https://github.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection)** 
8. [Fake News Detection 虚假新闻检测 (工具库)](https://github.com/ICTMCG/fake-news-detection)
9. [Fact Checking: Theory and Practice (KDD 2018 Tutorial)](https://shiralkarprashant.github.io/fact-checking-tutorial-KDD2018/)



