"""
@author: Lobster
@software: PyCharm
@file: model.py
@time: 2023/9/27 20:51
"""
import torch.nn


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # 文本编码 + 基础特征 + 一凡的文本编码 + 评论编码 + 评论三种情感度

        lstm_hidden_size = 1 + 23 + 1 + 1 + 1
        # lstm_hidden_size = 256

        # 进行一个长短期记忆操作
        self.lstm = torch.nn.LSTM(lstm_hidden_size, lstm_hidden_size)

        # 对微博原文编码的线性层

        self.text_linear1 = torch.nn.Linear(768, int(0.5 * 768))
        self.text_linear2 = torch.nn.Linear(int(0.5 * 768), 1)

        # 对一凡的原文 + comment联合编码的线性层

        self.text_linear3 = torch.nn.Linear(768, int(0.5 * 768))
        self.text_linear4 = torch.nn.Linear(int(0.5 * 768), 1)

        # 对comment编码的线性层

        self.text_linear5 = torch.nn.Linear(768, int(0.5 * 768))
        self.text_linear6 = torch.nn.Linear(int(0.5 * 768), 1)

        self.location_linear = torch.nn.Linear(10, 1)
        self.norm = torch.nn.BatchNorm1d(20 + 1 + 1)
        self.linear = torch.nn.Linear(1 + 23 + 1 + 1 + 1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text_embedding, verified, description, gender, messages, followers,
                location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval, has_url, emb,
                comment_embedding, positive, negative, neutral
                , tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate,
                likes_reposts_rate,total_at_num,reposts_tree_max_length,useful_comment_rate
                ):
        text_embedding = self.text_linear1(text_embedding)

        text_embedding = self.text_linear2(text_embedding)

        tree_length = reposts_tree_max_length * tree_avg_length

        emotion = torch.abs(positive - negative)

        emb = self.text_linear3(emb)

        emb = self.text_linear4(emb)

        comment_embedding = self.text_linear5(comment_embedding)

        comment_embedding = self.text_linear6(comment_embedding)

        norm_input = torch.cat([text_embedding,messages, followers,
                               location_embedding, friends, comments, pics, likes, reposts,
                               time_interval, emb, comment_embedding,emotion
                               ], dim=1)

        norm_input = self.norm(norm_input)

        mix_input = torch.cat([verified, description, gender,verified_type,has_url,norm_input
                               ], dim=1)

        lstm_output, _ = self.lstm(mix_input)

        output = self.linear(lstm_output)

        output = self.sigmoid(output)

        return output

# class Model(torch.nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#
#         # 文本编码 + 基础特征 + 一凡的文本编码 + 评论编码 + 评论三种情感度
#
#         lstm_hidden_size = 27
#         # lstm_hidden_size = 256
#
#         # 进行一个长短期记忆操作
#         self.lstm = torch.nn.LSTM(lstm_hidden_size, lstm_hidden_size)
#
#         # 对微博原文编码的线性层
#
#         self.text_linear1 = torch.nn.Linear(768, int(0.5 * 768))
#         self.text_linear2 = torch.nn.Linear(int(0.5 * 768), 1)
#
#         # 对一凡的原文 + comment联合编码的线性层
#
#         self.text_linear3 = torch.nn.Linear(768, int(0.5 * 768))
#         self.text_linear4 = torch.nn.Linear(int(0.5 * 768), 1)
#
#         # 对comment编码的线性层
#
#         self.text_linear5 = torch.nn.Linear(768, int(0.5 * 768))
#         self.text_linear6 = torch.nn.Linear(int(0.5 * 768), 40)
#
#         self.location_linear = torch.nn.Linear(10, 1)
#         self.norm = torch.nn.BatchNorm1d(22)
#         self.linear = torch.nn.Linear(27, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, text_embedding, verified, description, gender, messages, followers,
#                 location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval, has_url, emb,
#                 comment_embedding, positive, negative, neutral
#                 , tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate,
#                 likes_reposts_rate,total_at_num,reposts_tree_max_length,useful_comment_rate,
#                 ):
#         # text_embedding = self.text_linear1(text_embedding)
#         #
#         # text_embedding = self.text_linear2(text_embedding)
#
#         tree_length = reposts_tree_max_length * tree_avg_length
#
#         # emb = self.text_linear3(emb)
#         #
#         # emb = self.text_linear4(emb)
#
#         # comment_embedding = self.text_linear5(comment_embedding)
#         #
#         # comment_embedding = self.text_linear6(comment_embedding)
#
#         friends_followers_rate = friends / followers
#
#         likes_reposts_rate = likes / reposts
#
#         comments_reposts_rate = comments / reposts
#
#         norm_input = torch.cat([messages, followers,friends_followers_rate,likes_reposts_rate,comments_reposts_rate,
#                                location_embedding, friends, comments, pics, likes, reposts,
#                                time_interval,tree_length
#                                ], dim=1)
#
#         norm_input = self.norm(norm_input)
#
#         mix_input = torch.cat([verified, description, gender,verified_type,has_url,norm_input
#                                ], dim=1)
#
#         # mix_input = self.norm(mix_input)
#
#         lstm_output, _ = self.lstm(mix_input)
#
#         output = self.linear(lstm_output)
#
#         output = self.sigmoid(output)
#
#         return output

# def norm(features):
#     min_vals = torch.min(features, dim=0).values
#     max_vals = torch.max(features, dim=0).values
#
#     # 最大-最小归一化
#     normalized_features = (features - min_vals) / (max_vals - min_vals)
#
#     return normalized_features

# class Model(torch.nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#
#         # 文本编码 + 基础特征 + 一凡的文本编码 + 评论编码 + 评论三种情感度
#
#         lstm_hidden_size = 23
#         # lstm_hidden_size = 256
#
#         # 进行一个长短期记忆操作
#         self.lstm = torch.nn.LSTM(lstm_hidden_size, lstm_hidden_size)
#
#         # 对微博原文编码的线性层
#
#         self.text_linear1 = torch.nn.Linear(768 * 3, 768 * 2)
#
#         self.text_linear2 = torch.nn.Linear(768 * 2, 768)
#
#         self.text_linear3 = torch.nn.Linear(768, int(768 * 0.5))
#
#         self.text_linear4 = torch.nn.Linear(int(768 * 0.5), int(768 * 0.5 * 0.5))
#
#         self.text_linear5 = torch.nn.Linear(int(768 * 0.5 * 0.5), int(768 * 0.5 * 0.5 * 0.5))
#
#         self.text_linear6 = torch.nn.Linear(int(768 * 0.5 * 0.5 * 0.5), int(768 * 0.5 * 0.5 * 0.5 * 0.5))
#
#         self.text_linear7 = torch.nn.Linear(int(768 * 0.5 * 0.5 * 0.5 * 0.5), int(768 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5))
#
#         self.drop = torch.nn.Dropout(0.2)
#
#         self.location_linear = torch.nn.Linear(10, 1)
#
#         self.norm = torch.nn.BatchNorm1d(23)
#
#         self.linear = torch.nn.Linear(23, 1)
#
#         self.sigmoid = torch.nn.Sigmoid()
#
#         # self.softmax = torch.nn.Softmax()
#
#     def forward(self, text_embedding, verified, description, gender, messages, followers,
#                 location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval, has_url, emb,
#                 comment_embedding, positive, negative, neutral
#                 , tree_avg_length,is_positive,is_negative,friends_followers_rate,comments_reposts_rate,likes_reposts_rate
#                 ):
#         # text_mix_embedding = torch.cat([emb, comment_embedding, text_embedding], dim=1)
#         #
#         # text_mix_embedding = self.text_linear1(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear2(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear3(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear4(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear5(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear6(text_mix_embedding)
#         #
#         # text_mix_embedding = self.drop(text_mix_embedding)
#         #
#         # text_mix_embedding = self.text_linear7(text_mix_embedding)
#
#         mix_input = torch.cat([verified, description, gender, messages, followers,
#                                location_embedding, friends, verified_type, comments, pics, likes, reposts,
#                                time_interval, has_url
#                                ], dim=1)
#
#         # mix_input = torch.cat([verified, description, gender, messages,
#         #                        location_embedding, verified_type, pics,
#         #                        time_interval, has_url, friends_followers_rate,comments_reposts_rate,likes_reposts_rate
#         #                        ], dim=1)
#
#         mix_input = self.norm(mix_input)
#
#         lstm_output, _ = self.lstm(mix_input)
#
#         output = self.linear(lstm_output)
#
#         output = self.sigmoid(output)
#
#         return output
