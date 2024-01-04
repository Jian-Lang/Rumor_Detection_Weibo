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

        lstm_hidden_size = 1 + 23 + 1 + 1 + 15
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
        self.norm = torch.nn.BatchNorm1d(20 + 1 + 15)
        self.linear = torch.nn.Linear(1 + 23 + 1 + 1 + 15, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text_embedding, verified, description, gender, messages, followers,
                location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval, has_url, emb,
                comment_embedding, positive, negative, neutral
                , tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate,
                likes_reposts_rate, total_at_num, reposts_tree_max_length, useful_comment_rate, positive_max,
                negative_max, neutral_max, positive_original, negative_original, neutral_original
                ):
        text_embedding = self.text_linear1(text_embedding)

        text_embedding = self.text_linear2(text_embedding)

        tree_length = reposts_tree_max_length * tree_avg_length

        abs_minus_emotion = torch.abs(positive - negative)

        mean_emotion = torch.cat([positive, negative, neutral], dim=1)

        max_emotion = torch.cat([positive_max, negative_max, neutral_max], dim=1)

        emotion_1 = torch.cat([mean_emotion, max_emotion], dim=1)

        original_emotion = torch.cat([positive_original, negative_original, neutral_original], dim=1)

        dual_mean_emotion = original_emotion - mean_emotion

        dual_max_emotion = original_emotion - max_emotion

        emotion_2 = torch.cat([dual_mean_emotion, dual_max_emotion], dim=1)

        emotion = torch.cat([emotion_1, emotion_2, original_emotion], dim=1)

        emb = self.text_linear3(emb)

        emb = self.text_linear4(emb)

        comment_embedding = self.text_linear5(comment_embedding)

        comment_embedding = self.text_linear6(comment_embedding)

        norm_input = torch.cat([text_embedding, messages, followers,
                                location_embedding, friends, comments, pics, likes, reposts,
                                time_interval, emb, comment_embedding,emotion
                                ], dim=1)

        norm_input = self.norm(norm_input)

        mix_input = torch.cat([verified, description, gender, verified_type, has_url, norm_input
                               ], dim=1)

        lstm_output, _ = self.lstm(mix_input)

        output = self.linear(lstm_output)

        output = self.sigmoid(output)

        return output




    # def __init__(self, input_size):
    #     super().__init__()
    #
    #     # 文本编码 + 基础特征 + 一凡的文本编码 + 评论编码 + 评论三种情感度
    #
    #     lstm_hidden_size = 1 + 23 + 1 + 1 + 1
    #     # lstm_hidden_size = 256
    #
    #     # 进行一个长短期记忆操作
    #     self.lstm = torch.nn.LSTM(lstm_hidden_size, lstm_hidden_size)
    #
    #     # 对微博原文编码的线性层
    #
    #     self.text_linear1 = torch.nn.Linear(768, int(0.5 * 768))
    #     self.text_linear2 = torch.nn.Linear(int(0.5 * 768), 1)
    #
    #     # 对一凡的原文 + comment联合编码的线性层
    #
    #     self.text_linear3 = torch.nn.Linear(768, int(0.5 * 768))
    #     self.text_linear4 = torch.nn.Linear(int(0.5 * 768), 1)
    #
    #     # 对comment编码的线性层
    #
    #     self.text_linear5 = torch.nn.Linear(768, int(0.5 * 768))
    #     self.text_linear6 = torch.nn.Linear(int(0.5 * 768), 1)
    #
    #     self.location_linear = torch.nn.Linear(10, 1)
    #     self.norm = torch.nn.BatchNorm1d(20 + 1 + 1)
    #     self.linear = torch.nn.Linear(1 + 23 + 1 + 1 + 1, 1)
    #     self.sigmoid = torch.nn.Sigmoid()
    #
    # def forward(self, text_embedding, verified, description, gender, messages, followers,
    #             location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval, has_url, emb,
    #             comment_embedding, positive, negative, neutral
    #             , tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate,
    #             likes_reposts_rate,total_at_num,reposts_tree_max_length,useful_comment_rate,positive_max, negative_max, neutral_max, positive_original, negative_original, neutral_original
    #             ):
    #     text_embedding = self.text_linear1(text_embedding)
    #
    #     text_embedding = self.text_linear2(text_embedding)
    #
    #     tree_length = reposts_tree_max_length * tree_avg_length
    #
    #     emotion = torch.abs(positive - negative)
    #
    #     emb = self.text_linear3(emb)
    #
    #     emb = self.text_linear4(emb)
    #
    #     comment_embedding = self.text_linear5(comment_embedding)
    #
    #     comment_embedding = self.text_linear6(comment_embedding)
    #
    #     norm_input = torch.cat([text_embedding,messages, followers,
    #                            location_embedding, friends, comments, pics, likes, reposts,
    #                            time_interval, emb, comment_embedding,emotion
    #                            ], dim=1)
    #
    #     norm_input = self.norm(norm_input)
    #
    #     mix_input = torch.cat([verified, description, gender,verified_type,has_url,norm_input
    #                            ], dim=1)
    #
    #     lstm_output, _ = self.lstm(mix_input)
    #
    #     output = self.linear(lstm_output)
    #
    #     output = self.sigmoid(output)
    #
    #     return output

