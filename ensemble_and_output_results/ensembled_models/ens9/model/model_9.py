import torch
import torch.nn as nn


# 模型权重初始化函数
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(m.weight_hh, gain=nn.init.calculate_gain('relu'))


class RumorDetection(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RumorDetection, self).__init__()
        emb_size = 768

        self.text_transform = nn.Sequential(
            nn.Linear(emb_size, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

        self.emb_transform = nn.Sequential(
            nn.Linear(emb_size, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

        self.comment_transform = nn.Sequential(
            nn.Linear(emb_size, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

        self.loc_fc = nn.Linear(10, 1)

        self.norm = torch.nn.BatchNorm1d(14 + 10*3 + 1 + 1)
        self.linear = torch.nn.Linear(1 + 18, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.final_predict = nn.Sequential(
            nn.Linear(14 + 10*3 + 1 + 1 + 1, 10),
            nn.ReLU(),
            nn.Linear(10, out_dim),
        )

        # self.apply(weight_init)


    def forward(self, text_embedding, verified, messages, followers,
                location_embedding, friends, comments, likes, reposts,
                emb, comment_embedding, positive, negative, neutral
                , tree_avg_length, is_positive, is_negative, friends_followers_rate,
                total_at_num, reposts_tree_max_length, useful_comment_rate,
                ):

        location_embedding = self.loc_fc(location_embedding)
        text_embedding = self.text_transform(text_embedding)
        emb = self.emb_transform(emb)
        comment_embedding = self.comment_transform(comment_embedding)
        emotion = torch.abs(positive - negative)

        norm_input = torch.cat([messages, followers, friends, comments, likes, reposts,
                                positive, negative, neutral, tree_avg_length, friends_followers_rate, total_at_num,
                                reposts_tree_max_length, useful_comment_rate,
                                text_embedding,
                                emb,
                                comment_embedding,
                                location_embedding,emotion
                                ], dim=1)

        norm_input = self.norm(norm_input)

        output = self.final_predict(torch.cat([norm_input, verified], dim=1))

        output = self.sigmoid(output)

        return output


