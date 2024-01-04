"""
@author: Lobster
@software: PyCharm
@file: train.py
@time: 2023/11/23 11:20
"""

import logging
import os
import sys
import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataLoader.dataset import MyData,custom_collate_fn
from model.base_model import Model
# from data.data_prepocessing.text_encoding import text_encoding

# 文本颜色设置

BLUE = '\033[94m'
ENDC = '\033[0m'


def print_init_msg(logger, args):

    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    # logger.info(BLUE + "Learning Decay: " + ENDC + f"{args.decay_rate}")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):
    # 创建文件夹和日志文件，用于记录训练结果和模型

    # 获取当前时间戳

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"

    # 指定文件夹的完整路径

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    # 创建文件夹

    os.mkdir(folder_path)

    os.mkdir(os.path.join(folder_path, "trained_model"))

    # 配置日志记录

    # 创建logger对象

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # 创建控制台处理器

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    # 创建文件处理器

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    # 设置日志格式

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    # 将处理器添加到logger对象中

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn_list):

    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")

    best_model_name_list = []

    for i in range(len(min_turn_list)):

        best_model_name_list.append(f'model_{i + 1}_{min_turn_list[i]}.pth')

    for i in range(len(model_name_list)):

        if model_name_list[i] not in best_model_name_list:

            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):

    print(msg)

    sys.exit(1)


def train_val(args):
    # 通过args解析出所有参数

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    # device

    device = torch.device(args.device)

    # 加载数据集

    df = pd.read_pickle(args.dataset_path)

    # # 加载模型
    #
    model = Model(input_size=24)

    model = model.to(device)

    # 定义损失函数

    if args.loss == 'BCE':

        loss_fn = torch.nn.BCELoss()

    elif args.loss == 'MSE':

        loss_fn = torch.nn.MSELoss()

    else:

        force_stop('Invalid parameter loss!')

    loss_fn.to(device)

    # # 定义优化器
    #
    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    # 定义学习率衰减

    # decayRate = args.decay_rate

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decayRate)

    # 定义训练过程的一些参数

    min_total_valid_loss = 1008611

    min_counter_valid_loss = 1008611

    min_turn = 0

    min_counter = 0

    # 开始训练

    print_init_msg(logger, args)

    ######################## 十折训练法 ##############################

    # 设置折数

    num_folds = 10

    # 创建 KFold 对象

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=918)

    # 存储每折的性能指标

    min_turn_list = []

    # 开始十折交叉验证

    counter = 0

    for fold, (train_index, valid_index) in enumerate(kf.split(df)):

        # 加载模型

        model = Model(input_size=24)

        model = model.to(device)

        if args.optim == 'Adam':

            optim = Adam(model.parameters(), args.lr)

        elif args.optim == 'SGD':

            optim = SGD(model.parameters(), args.lr)

        else:

            force_stop('Invalid parameter optim!')

        # 通过十折，获取训练集和测试集

        logger.info(f"-----------------------------------Counter {counter + 1} Start!-----------------------------------")

        min_total_valid_loss = 1008611

        train_data, valid_data = df.iloc[train_index], df.iloc[valid_index]

        train_data.reset_index(drop=True, inplace=True)

        valid_data.reset_index(drop=True, inplace=True)

        train_data = MyData(train_data)

        valid_data = MyData(valid_data)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

        for i in range(args.epochs):

            logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

            min_train_loss, total_valid_loss = run_one_epoch(model, loss_fn, optim, train_data_loader,
                                                             valid_data_loader,
                                                             device)

            # scheduler.step()

            logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")

            logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

            if total_valid_loss < min_total_valid_loss:

                min_total_valid_loss = total_valid_loss

                min_turn = i + 1

            logger.critical(
                f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

            torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{counter + 1}_{i + 1}.pth")

            logger.info("Model has been saved successfully!")

            if (i + 1) - min_turn > args.early_stop_turns:
                break

        min_turn_list.append(min_turn)

        if min_total_valid_loss < min_counter_valid_loss:
            min_counter_valid_loss = min_total_valid_loss
            min_counter = counter

        logger.critical(f"Current Best Total Loss comes from Counter {min_counter + 1} , min_total_loss = {min_counter_valid_loss}")

        counter += 1

    delete_model(father_folder_name, folder_name, min_turn_list)

    logger.info(BLUE + "Training is ended!" + ENDC)


def run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    # 训练部分

    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

        id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
            verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive_avg, \
            negative_avg, neutral_avg, reposts_tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate, \
            total_at_num,reposts_tree_max_length,useful_comment_rate,positive_max,negative_max,neutral_max,positive_original,negative_original,neutral_original,word_rumor_score_base,label = batch

        output = model(text_embedding, verified, description, gender, messages, followers,
                       location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval,
                       has_url, emb, comment_embedding, positive_avg, negative_avg, neutral_avg,
                       reposts_tree_avg_length, is_positive,
                       is_negative,friends_followers_rate,comments_reposts_rate,likes_reposts_rate,total_at_num,reposts_tree_max_length,useful_comment_rate
                       ,positive_max,negative_max,neutral_max,positive_original,negative_original,neutral_original,word_rumor_score_base)

        loss = loss_fn(output, label)

        # 通过损失，优化参数

        optim.zero_grad()

        loss.backward()

        optim.step()

        if min_train_loss > loss:

            min_train_loss = loss

    # 验证环节

    model.eval()

    total_valid_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
                verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive_avg, \
                negative_avg, neutral_avg, reposts_tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate, \
                total_at_num, reposts_tree_max_length, useful_comment_rate, positive_max, negative_max, neutral_max, positive_original, negative_original, neutral_original,word_rumor_score_base, label = batch

            output = model(text_embedding, verified, description, gender, messages, followers,
                           location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval,
                           has_url, emb, comment_embedding, positive_avg, negative_avg, neutral_avg,
                           reposts_tree_avg_length, is_positive,
                           is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate, total_at_num,
                           reposts_tree_max_length, useful_comment_rate
                           , positive_max, negative_max, neutral_max, positive_original, negative_original,
                           neutral_original,word_rumor_score_base)

            # id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
            #     verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, label = batch
            #
            # output = model(text_embedding, verified, description, gender, messages, followers,
            #                location_embedding, friends, verified_type, comments, pics, likes, reposts, time_interval,
            #                has_url, emb, comment_embedding)

            loss = loss_fn(output, label)

            total_valid_loss += loss

    return min_train_loss, total_valid_loss


# 主函数，所有训练参数在这里调整

def main():
    # 创建一个ArgumentParser对象

    parser = argparse.ArgumentParser()

    # 运行前命令行参数设置

    parser.add_argument('--device', default='cuda:0', type=str, help='device used in training')

    parser.add_argument('--metric', default='BCE', type=str, help='the judgement of the training')

    parser.add_argument('--save', default='train_results', type=str, help='folder to save the results')

    parser.add_argument('--epochs', default=2000, type=int, help='max number of training epochs')

    # 注意，大的batch_size，梯度比较平滑，可以设置大的learning rate，vice visa

    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')

    parser.add_argument('--early_stop_turns', default=100, type=int, help='early stop turns of training')

    parser.add_argument('--loss', default='BCE', type=str, help='loss function, options: BCE, MSE')

    parser.add_argument('--optim', default='Adam', type=str, help='optim, options: SGD, Adam')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    # trick: 先按1.0，即不decay学习率训练，震荡不收敛，可以适当下调

    # parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')

    parser.add_argument('--dataset_id', default='rumor_train', type=str, help='id of dataset')

    # 提交的pkl，用train_with_user.pkl

    parser.add_argument('--dataset_path', default=r'D:\RumorDetection\data\train_data\train_with_user.pkl', type=str, help='path of dataset')

    parser.add_argument('--model_id', default='mlp', type=str, help='id of model')

    args = parser.parse_args()

    train_val(args)


if __name__ == '__main__':
    main()
