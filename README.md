# NLP情感分类实践

[TOC]

## 情感分析简介

情感分析即**利用算法来分析提取文本中表达的情感**。分析一个句子表达的好、中、坏等判断，高兴、悲伤、愤怒等情绪。如果能将这种文字转为情感的操作让计算机自动完成，就节省了大量的时间。对于目前的海量文本数据来说，这是很有必要的。
我们可以通过情感分析，在电商领域挖掘出口碑好的商品，订餐订住宿领域挖掘优质场所等等。

文本情感分析主要有三大任务 即文本情感特征提取，**文本情感特征分类**，文本情感特征检索与归纳。本文的实战主要是关于情感特征分类。

## 数据

语料数据来源于电影评论：

* 训练集合20000（正向10000，负向10000）
* 测试集合20000（正向3000，负向3000）

* 语料处理：使用jieba进行分词

数据示例：

```txt
1	死囚 爱 刽子手 女贼 爱 衙役 我们 爱 你们 难道 还有 别的 选择 没想到 ...
1	其实 我 对 锦衣卫 爱情 很萌 因为 很 言情小说 可惜 女主角 我要 不是 ...
1	两星 半 小 明星 本色 出演 老 演员 自己 发挥 基本上 王力宏 表演 ...
......
0	蒂姆 波顿 简直 太棒 拍 出来 哥特 风格 像 画 一样 JD 表演 风格 应该 ...
0	又 看上去 扯蛋 青春 情色 最后 扯 出 人性 政治 时代 思想 形态 片子 ...
0	部 忠于 原著 电影 出于 时间 原因 做 人物 调整 对于 多分钟 一部 电影 ...
```

### 预处理

* 去除停用词
* word2id和id2word生成
* 词向量生成：使用预训练的word2vec向量
* 句子padding，使得批处理时长度一致

## 模型

### encoder层

* BiLSTM+Attention：
  * num_layers: 2
  * attention方式: concat
* BiLSTM:
  * num_layers: 2

```python
    def __init__(self, vocab_size, embedding_dim,pretrained_weight, update_w2v,hidden_dim,
                 num_layers,drop_keep_prob,n_class,bidirectional, **kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=drop_keep_prob)
```

### classifier层

* 两个线性层，映射到num_classes的维度

```python
self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
self.decoder2 = nn.Linear(hidden_dim,n_class)
```

### attention部分

公式：
$att = v_a^{T}\tanh(W_a[h_t;h_s])$


```python
    def forward(self, inputs):
        embeddings = self.embedding(inputs)# [batch, seq_len] => [batch, seq_len, embed_dim][64,75,50]

        states, hidden = self.encoder(embeddings.permute([0, 1, 2]))#[batch, seq_len, embed_dim]
        #attention
        # states:(seq_len, batch, num_directions*hidden_zie)
        u = torch.tanh(torch.matmul(states, self.weight_W))
        att = torch.matmul(u, self.weight_proj)

        att_score = F.softmax(att, dim=1)
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        outputs = self.decoder1(encoding)
        outputs=self.decoder2(outputs)
        return outputs 
```

## 训练

训练参数：

```python
update_w2v = True          # 是否在训练中更新w2v
vocab_size = 54848          # 词汇量，与word2id中的词汇量一致
n_class = 2                 # 分类数：分别为pos和neg
max_sen_len = 65           # 句子最大长度
embedding_dim = 50          # 词向量维度
batch_size =64            # 批处理尺寸
hidden_dim=100           # 隐藏层节点数
n_epoch = 30            # 训练迭代周期，即遍历整个训练样本的次数
lr = 0.0001               # 学习率；若opt=‘adadelta'，则不需要定义学习率
drop_keep_prob = 0.2        # dropout层，参数keep的比例
num_layers = 2              # LSTM层数
bidirectional=True         #是否使用双向LSTM
```

* step1: 定义dataloader

  ```python
      train_loader = Data_set(train_array, train_lable)
      train_dataloader = DataLoader(train_loader,
                                   batch_size=Config.batch_size,
                                   shuffle=True,
                                   num_workers=0)#用了workers反而变慢了
  ```

* step2: 加载预训练词向量权重

  ```
      w2vec=build_word2vec(Config.pre_word2vec_path,word2id,None)#生成word2vec
      w2vec=torch.from_numpy(w2vec)
  ```

* step3: 初始化model

  ```python
  if Config.attention:
  	model = LSTM_attention(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
  												 Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,  															 Config.bidirectional)
  else:
  	model = LSTMModel(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
  										Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class, 																Config.bidirectional)
  ```

* step4: 定义优化器optimizer、loss

  ```python
      optimizer = optim.Adam(model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss()
  ```

* step5: 模型前馈计算，loss反向传播，更新梯度, 框架如下

  ```python
      for epoch in range(epoches):  # 一个epoch可以认为是一次训练循环
      		...
          train_dataloader = tqdm.tqdm(train_dataloader)
          for i, data_ in (enumerate(train_dataloader)):
              optimizer.zero_grad()
              ...
              output = model(input_)
              # 经过模型对象就产生了输出
              ...
              loss = criterion(output, target)
              loss.backward()
              optimizer.step()
              train_loss += loss.item()
              _, predicted = torch.max(output, 1)
              ...
              # 在验验证集上计算模型准确率，并保存最优模型
              acc = val_accuary(model, val_dataloader, device, criterion)
              if acc > best_acc:
                  best_acc = acc
                  if os.path.exists(Config.model_state_dict_path) == False:
                      os.mkdir(Config.model_state_dict_path)
                  torch.save(model, '...')  # 这种方式把整个模型都存了下来
  ```

## 预测

* step1: torch.load()加载模型
* step2: 模型前馈计算得到输出，取概率最大的index作为预测label

```python
def predict(word2id,model,seq_lenth ,path):
    model.cpu()
    with torch.no_grad():
        input_array=text_to_array_nolable(word2id,seq_lenth,path)
        #sen_p = sen_p.type(torch.LongTensor)
        sen_p = torch.from_numpy(input_array)
        sen_p=sen_p.type(torch.LongTensor)
        output_p = model(sen_p)
        _, pred = torch.max(output_p, 1)
        for i in pred:
            print('预测类别为',i.item())
```

## 评价指标

常用的评价指标：

* accuracy：[sklearn.metrics.accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html?highlight=accuracy_score#sklearn-metrics-accuracy-score)
* F1_score:   [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html?highlight=f1_score#sklearn-metrics-f1-score)
* recall：[sklearn.metrics.recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html?highlight=recall#sklearn.metrics.recall_score)
* precision：[sklearn.metrics.precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html?highlight=precision#sklearn.metrics.precision_score)

## 结果

| 模型             | accuracy | f1_score | recall |
| ---------------- | -------- | -------- | ------ |
| BiLSTM+Attention | 0.84     |          |        |
| BiLSTM           |          |          |        |



## 附

* 常用数据集/API

以下是一些我们最常用的的用于试验情绪分析和机器学习方法的情绪分析数据集。 它们开源且可以免费下载: •产品评论:此数据集包含数百万亚马逊客户评论，星级评级，对培训情绪分析模型非常有用;
 ( https://www.kaggle.com/bittlingmayer/amazonreviews)

•餐饮评论:此数据集包含5,2百万条评论星级的Yelp评论;( https://www.kaggle.com/yelp-dataset/yelp-dataset) •电影评论:此数据集包含1,000个正面和1,000个负面处理评论。 它还提供5,331个正面和5,331个负面处理句子/片段;
 ( http://www.cs.cornell.edu/people/pabo/movie-review-data/) •精美食品评论:此数据集包含来自亚马逊的约500,000份食品评论。 它包括产品和用户信息，评级以及每个评论的纯文本版 本;( https://www.kaggle.com/snap/amazon-fine-food-reviews) •Kaggle上Twitter对航空公司的评论:该数据集包含约15,000条有关航空公司的标签推文(正面，中性和负面);
 ( https://www.kaggle.com/crowdflower/twitter-airline-sentiment) •第一次共和党辩论Twitter情绪:这个数据集由关于2016年第一次共和党辩论的大约14,000条标记的推文(正面，中立和负面) 组成。( https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)



* Bi-LSTM+attention. Pipeline

1. 拿到文本，分词，清洗数据(去掉停用词语)

2. 建立word2index index2word 表

3. 准备好预训练好的 word embedding ( or start from one hot)

4. 做好 Dataset / Dataloader

5. 建立模型. (soft attention. / hard attention/ self-attention/ scaled dot

   product self attention)

6. 配置好参数

7. 开始训练

8. 测评

9. 保存模型

## Reference

1. 文本情感分析方法小结 **LeonG** ( https://zhuanlan.zhihu.com/p/106588589) 
2. **Chineses-Sentiment Analysis-Pytorch** :李狗嗨 (https://github.com/Ligouhai-bigone/Sentiment-Analysis-Chinese-pytorch)
3. **Recall . Precision**(https://www.zhihu.com/question/19645541/answer/91694636)
4. **Bilstm+attention**(https://www.cnblogs.com/jiangxinyang/p/10208227.html) 
5. **Lstm**(https://zybuluo.com/hanbingtao/note/581764)

