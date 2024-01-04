"""
@author: Lobster
@software: PyCharm
@file: dataset.py
@time: 2023/9/27 20:51
"""
import torch.utils.data
import pandas as pd
from sklearn.model_selection import KFold


def custom_collate_fn(batch):
    # id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
    #     verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding,label = zip(*batch)

    id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
        verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive_avg, \
        negative_avg, neutral_avg, reposts_tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate,\
        total_at_num,reposts_tree_max_length,useful_comment_rate,positive_max,negative_max,neutral_max,positive_original,negative_original,neutral_original,word_rumor_score_base,sentence_similarity,label = zip(*batch)

    # 返回一个包含三个张量的元组

    return list(id), list(text), list(uid), \
        torch.tensor(verified, dtype=torch.float32).view(-1, 1), torch.tensor(description, dtype=torch.float32).view(-1,
                                                                                                                     1), \
        torch.tensor(gender, dtype=torch.float32).view(-1, 1), torch.tensor(messages, dtype=torch.float32).view(-1,
                                                                                                                1), torch.tensor(
        followers, dtype=torch.float32).view(-1, 1), \
        torch.tensor(location_embedding, dtype=torch.float32), torch.tensor(reg_time, dtype=torch.float32).view(-1,
                                                                                                                1), torch.tensor(
        friends, dtype=torch.float32).view(-1, 1), \
        torch.tensor(verified_type, dtype=torch.float32).view(-1, 1), torch.tensor(has_url, dtype=torch.float32).view(
        -1, 1), torch.tensor(comments, dtype=torch.float32).view(-1, 1), \
        torch.tensor(pics, dtype=torch.float32).view(-1, 1), torch.tensor(likes, dtype=torch.float32).view(-1, 1), \
        torch.tensor(time, dtype=torch.float32).view(-1, 1), \
        torch.tensor(reposts, dtype=torch.float32).view(-1, 1), list(mid), torch.tensor(time_interval,
                                                                                        dtype=torch.float32).view(-1,
                                                                                                                  1), \
        torch.tensor(text_embedding, dtype=torch.float32), torch.tensor(emb, dtype=torch.float32), torch.tensor(
        comment_embedding, dtype=torch.float32), torch.tensor(positive_avg,dtype=torch.float32).view(-1, 1),torch.tensor(negative_avg,dtype=torch.float32).view(-1, 1),\
        torch.tensor(neutral_avg,dtype=torch.float32).view(-1, 1),torch.tensor(reposts_tree_avg_length,dtype=torch.float32).view(-1,1),\
        torch.tensor(is_positive,dtype=torch.float32).view(-1,1), torch.tensor(is_negative,dtype=torch.float32).view(-1,1), \
        torch.tensor(friends_followers_rate,dtype=torch.float32).view(-1, 1), torch.tensor(comments_reposts_rate,dtype=torch.float32).view(-1,1), \
        torch.tensor(likes_reposts_rate,dtype=torch.float32).view(-1,1),torch.tensor(total_at_num,dtype=torch.float32).view(-1,1),\
        torch.tensor(reposts_tree_max_length,dtype=torch.float32).view(-1,1), torch.tensor(useful_comment_rate, dtype=torch.float32).view(-1,1), \
        torch.tensor(positive_max,dtype=torch.float32).view(-1,1), torch.tensor(negative_max,dtype=torch.float32).view(-1,1),\
        torch.tensor(neutral_max,dtype=torch.float32).view(-1,1), torch.tensor(positive_original,dtype=torch.float32).view(-1,1),torch.tensor(negative_original,dtype=torch.float32).view(-1,1), \
        torch.tensor(neutral_original,dtype=torch.float32).view(-1,1),torch.tensor(word_rumor_score_base,dtype=torch.float32).view(-1,1),torch.tensor(sentence_similarity,dtype=torch.float32).view(-1,1),\
        torch.tensor(label, dtype=torch.float32).view(-1, 1)


class MyData(torch.utils.data.Dataset):

    def __init__(self, dataframe):
        super().__init__()

        # 这部分相当于是加载一些基础的数据，来自text_data或train_data

        self.dataframe = dataframe

        # 加载公共的数据，即微博原文数据、微博评论数据等

        origin_text_df = pd.read_pickle(r'D:\RumorDetection\data\common_data\origin_text.pkl')

        comment_df = pd.read_pickle(r'D:\RumorDetection\data\common_data\comment.pkl')

        reposts_tree_df = pd.read_pickle(r'D:\RumorDetection\data\common_data\reposts.pkl')

        # 原始数据与公共数据进行拼接

        self.dataframe = pd.merge(self.dataframe, origin_text_df, on='id', how='inner')

        self.dataframe = pd.merge(self.dataframe, comment_df, on='id', how='inner')

        self.dataframe = pd.merge(self.dataframe, reposts_tree_df, on='id', how='inner')

        # 下面是各部分的属性列

        # 微博原文相关的features，即

        self.id_list = self.dataframe['id']

        self.text_list = self.dataframe['text']

        self.uid_list = self.dataframe['uid']

        self.verified_list = self.dataframe['verified']

        self.description_list = self.dataframe['description']

        self.gender_list = self.dataframe['gender']

        # 换归一化结果

        self.messages_list = self.dataframe['messages']

        # self.messages_list = self.dataframe['messages_norm']

        # 换归一化结果

        self.followers_list = self.dataframe['followers']

        # self.followers_list = self.dataframe['followers_norm']

        self.location_embedding_list = self.dataframe['location_embedding']

        # 换归一化结果

        self.reg_time_list = self.dataframe['reg_time']

        # self.reg_time_list = self.dataframe['reg_time_norm']

        # 换归一化结果

        self.friends_list = self.dataframe['friends']

        # self.friends_list = self.dataframe['friends_norm']

        self.verified_type_list = self.dataframe['verified_type']

        self.has_url_list = self.dataframe['has_url']

        # 换归一化结果

        self.comments_list = self.dataframe['comments']

        # self.comments_list = self.dataframe['comments_norm']

        # 换归一化结果

        self.pics_list = self.dataframe['pics']

        # self.pics_list = self.dataframe['pics_norm']

        # 换归一化结果

        self.likes_list = self.dataframe['likes']

        # self.likes_list = self.dataframe['likes_norm']

        # 换归一化结果

        self.time_list = self.dataframe['time']

        # self.time_list = self.dataframe['time_norm']

        # 换归一化结果

        self.reposts_list = self.dataframe['reposts']

        # self.reposts_list = self.dataframe['reposts_norm']

        self.mid_list = self.dataframe['mid']

        self.label_list = self.dataframe['label']

        # 特征工程部分

        self.time_interval_list = self.dataframe['time_interval_norm']

        self.text_embedding_list = self.dataframe['bert_base_chinese_embedding_mean_pooling']

        self.emb_list = self.dataframe['emb']

        self.comment_embedding_list = self.dataframe['comment_embedding_chinese_bert']

        self.positive_avg_list = self.dataframe['positive_avg']

        self.negative_avg_list = self.dataframe['negative_avg']

        self.neutral_avg_list = self.dataframe['neutral_avg']

        self.reposts_tree_avg_length_list = self.dataframe['avg_length']

        self.is_positive_list = self.dataframe['is_positive']

        self.is_negative_list = self.dataframe['is_negative']

        self.friends_followers_rate_list = self.dataframe['friends_followers_rate']

        self.comments_reposts_rate_list = self.dataframe['comments_reposts_rate']

        self.likes_reposts_rate_list = self.dataframe['likes_reposts_rate']

        self.total_at_num_list = self.dataframe['total_@_num']

        self.reposts_tree_max_length_list = self.dataframe['max_length']

        self.useful_comment_rate_list = self.dataframe['useful_comment_rate']

        self.comment_positive_list = self.dataframe['positive_list']

        # 情感分析更进一步

        self.positive_max_list = self.dataframe['positive_max']

        self.negative_max_list = self.dataframe['negative_max']

        self.neutral_max_list = self.dataframe['neutral_max']

        # 这是原文的情感度

        self.positive_list = self.dataframe['positive']

        self.negative_list = self.dataframe['negative']

        self.neutral_list = self.dataframe['neutral']

        # 词级别的特征工程

        self.word_rumor_score_base_list = self.dataframe['word_rumor_score_base']

        self.sentence_similarity_list = self.dataframe['sentence_similarity']

    def __getitem__(self, item):
        # 文件名，唯一关联几个数据集的标识

        id = self.id_list[item]

        #####################################

        # 第一部分数据：基础数据，来自微博原文的特征

        # 微博原文内容

        text = self.text_list[item]

        # 源用户，即发博人的id

        uid = self.uid_list[item]

        # 发博人的性别：有用

        gender = self.gender_list[item]

        # 发博人是否认证：bool：有用

        verified = self.verified_list[item]

        # 发博人的微博主页是否有个人描述：bool：有

        description = self.description_list[item]

        # 发博人的发文总数：有

        messages = self.messages_list[item]

        # 发博人的粉丝数：有

        followers = self.followers_list[item]

        # # 发博人的注册地址：暂定有用
        #
        # location = self.location_list[item]

        # 发博人的微博注册时间：有用

        reg_time = self.reg_time_list[item]

        # 发博人的关注他人的数量：暂定有用

        friends = self.friends_list[item]

        # 发博人的认证类型：有用

        verified_type = self.verified_type_list[item]

        # 发博人的这篇文章是否包含url：bool：暂定没用

        has_url = self.has_url_list[item]

        # 发博人的这篇发文的评论数：有用

        comments = self.comments_list[item]

        # 发博人的这篇文章包含的图片数：暂定有用

        pics = self.pics_list[item]

        # 发博人的这篇文章的点赞数：有用

        likes = self.likes_list[item]

        # 发博人的这篇文章的转发数：有用

        reposts = self.reposts_list[item]

        # 发博人的这篇文章的发文时间：有用，结合注册时间

        time = self.time_list[item]

        # 发博人的这篇文章的发文id

        mid = self.mid_list[item]

        #####################################

        # 第二部分数据：

        ####################################

        # 第三部分数据：特征工程

        time_interval = time - reg_time

        text_embedding = self.text_embedding_list[item]

        location_embedding = self.location_embedding_list[item]

        label = self.label_list[item]

        # 一凡的编码：comment + text

        emb = self.emb_list[item]

        comment_embedding = self.comment_embedding_list[item]

        positive_avg = self.positive_avg_list[item]

        negative_avg = self.negative_avg_list[item]

        neutral_avg = self.neutral_avg_list[item]

        reposts_tree_avg_length = self.reposts_tree_avg_length_list[item]

        is_positive = self.is_positive_list[item]

        is_negative = self.is_negative_list[item]

        friends_followers_rate = self.friends_followers_rate_list[item]

        comments_reposts_rate = self.comments_reposts_rate_list[item]

        likes_reposts_rate = self.likes_reposts_rate_list[item]

        total_at_num = self.total_at_num_list[item]

        reposts_tree_max_length = self.reposts_tree_max_length_list[item]

        useful_comment_rate = self.useful_comment_rate_list[item]

        positive_max = self.positive_max_list[item]

        negative_max = self.negative_max_list[item]

        neutral_max = self.neutral_max_list[item]

        positive_original = self.positive_list[item]

        negative_original = self.negative_list[item]

        neutral_original = self.neutral_list[item]

        word_rumor_score_base = self.word_rumor_score_base_list[item]

        sentence_similarity = self.sentence_similarity_list[item]

        return id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
            verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive_avg, \
            negative_avg, neutral_avg, reposts_tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate,\
            total_at_num,reposts_tree_max_length,useful_comment_rate,positive_max,negative_max,neutral_max,positive_original,negative_original,neutral_original,word_rumor_score_base,sentence_similarity,label

        # return id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
        #     verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, label

    def __len__(self):
        return len(self.dataframe)


if __name__ == "__main__":

    # 设置折数

    num_folds = 5

    # 创建 KFold 对象

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # 存储每折的性能指标

    fold_metrics = []

    # 把数据加载进来

    df = pd.read_pickle(r'D:\RumorDetection\data\train.pkl')

    # 开始十折交叉验证

    for fold, (train_index, test_index) in enumerate(kf.split(df)):

        # 获取训练集和测试集

        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        train_data.reset_index(drop=True, inplace=True)

        test_data.reset_index(drop=True, inplace=True)

        train_data = MyData(train_data)

        test_data = MyData(test_data)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64)

        for batch in train_data_loader:
            id, text, uid, verified, description, gender, messages, followers, location, reg_time, friends, \
                verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, label = batch
