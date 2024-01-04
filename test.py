import numpy as np
import pandas as pd

"""
@author: Lobster
@software: PyCharm
@file: test.py
@time: 2023/9/27 20:51
"""
import argparse
import os
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataLoader.dataset import MyData, custom_collate_fn
from model.base_model import Model

# 文本颜色设置

BLUE = '\033[94m'
ENDC = '\033[0m'


def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def make_father_folder(args):
    # model id

    model_id = args.model_id

    # dataset id

    dataset_id = args.dataset_id

    # metric

    metric = args.metric

    # 创建文件夹和日志文件，用于记录验证集的结果

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称
    folder_name = f"test_{model_id}_{dataset_id}_{metric}_{timestamp}"

    # 指定文件夹的完整路径

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    # 创建文件夹
    os.mkdir(folder_path)

    return father_folder_name, folder_name


def make_logger(father_folder_name, folder_name):
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

    return logger


def test(args, model_path, logger):
    # model id

    model_id = args.model_id

    # dataset id

    dataset_id = args.dataset_id

    # metric

    metric = args.metric

    # device

    device = torch.device(args.device)

    # 加载数据集

    batch_size = args.batch_size

    test_data = MyData(pd.read_pickle(args.dataset_path))

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=custom_collate_fn)

    # 加载训练好的模型, 这里记得import你的模型

    model = torch.load(model_path)

    # 定义验证相关参数

    total_test_step = 0

    total_f1 = 0

    # 开始验证

    # logger.info(BLUE + 'Device: ' + ENDC + f"{device} ")
    #
    # logger.info(BLUE + 'Model: ' + ENDC + f"{model_id} ")
    #
    # logger.info(BLUE + "Dataset: " + ENDC + f"{dataset_id}")
    #
    # logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    #
    # logger.info(BLUE + "Training Starts!" + ENDC)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
                verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive, \
                negative, neutral, tree_avg_length, is_positive, is_negative, label = batch

            output = model(text_embedding, verified, description, gender, messages, followers,
                           location_embedding, friends, verified_type, comments, pics, likes, reposts,
                           time_interval,
                           has_url, emb, comment_embedding, positive, negative, neutral, tree_avg_length, is_positive,
                           is_negative)

            output = output.to('cpu')

            label = label.to('cpu')

            threshold = 0.5  # 设置阈值

            binary_output = (output > threshold).int().tolist()  # 将概率值大于阈值的设为1，否则为0

            f1 = f1_score(label, binary_output)

            total_test_step += 1

            total_f1 += f1

    logger.warning(f"path : {model_path}")

    logger.warning(f"[ Test Result ]:  f1 = {total_f1 / total_test_step}")

    logger.info("Test is ended!")


def test_ens(args, model_list, logger):
    # model id

    model_id = args.model_id

    # dataset id

    dataset_id = args.dataset_id

    # metric

    metric = args.metric

    # device

    device = torch.device(args.device)

    # 加载数据集

    batch_size = args.batch_size

    test_data = MyData(pd.read_pickle(args.dataset_path))

    test_data_loader = DataLoader(dataset=test_data, batch_size=len(pd.read_pickle(args.dataset_path)),
                                  collate_fn=custom_collate_fn)

    # 加载训练好的模型, 这里记得import你的模型

    # model = torch.load(model_path)

    # 定义验证相关参数

    total_test_step = 0

    total_f1 = 0

    total_output = 0

    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Testing'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            output_list = []

            id, text, uid, verified, description, gender, messages, followers, location_embedding, reg_time, friends, \
                verified_type, has_url, comments, pics, likes, time, reposts, mid, time_interval, text_embedding, emb, comment_embedding, positive_avg, \
                negative_avg, neutral_avg, reposts_tree_avg_length, is_positive, is_negative, friends_followers_rate, comments_reposts_rate, likes_reposts_rate, \
                total_at_num, reposts_tree_max_length, useful_comment_rate, positive_max, negative_max, neutral_max, positive_original, negative_original, neutral_original,label = batch

            for i in range(len(model_list)):
                model = torch.load(model_list[i])
                model.eval()
                output = model(text_embedding, verified, description, gender, messages, followers,
                               location_embedding, friends, verified_type, comments, pics, likes, reposts,
                               time_interval,
                               has_url, emb, comment_embedding, positive_avg, negative_avg, neutral_avg,
                               reposts_tree_avg_length, is_positive, is_negative,
                               friends_followers_rate, comments_reposts_rate, likes_reposts_rate, total_at_num,
                               reposts_tree_max_length, useful_comment_rate,positive_max, negative_max, neutral_max, positive_original, negative_original, neutral_original)
                if i == 0:
                    total_output = output
                else:
                    total_output += output

            output = total_output / 10

            output = output.to('cpu')

            label = label.to('cpu')

            threshold = 0.5  # 设置阈值

            binary_output = (output > threshold).int().tolist()  # 将概率值大于阈值的设为1，否则为0

            f1 = f1_score(label,binary_output)

            total_test_step += 1

            total_f1 += f1

    # logger.warning(f"path : {model_path}")

    logger.warning(f"[ Test Result ]:  f1 = {total_f1 / total_test_step}")

    logger.info("Test is ended!")

    return output,id


def main():
    # 创建一个ArgumentParser对象

    parser = argparse.ArgumentParser()

    # 运行前命令行参数设置

    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')

    parser.add_argument('--metric', default='BCE', type=str, help='the judgement of the training')

    parser.add_argument('--save', default='test_results', type=str, help='folder to save the results')

    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')

    parser.add_argument('--dataset_id', default='rumor', type=str, help='id of dataset')

    parser.add_argument('--dataset_path', default=r'D:\RumorDetection\data\train_data\train_with_user_test.pkl',
                        type=str, help='path of dataset folder')

    parser.add_argument('--model_id', default='mlp', type=str, help='id of model')

    parser.add_argument('--model_path',
                        default=r'D:\RumorDetection\train_results\train_mlp_rumor_train_BCE_2023-11-25_17-47-14\trained_model\model_7_1.pth',
                        type=str, help='path of trained model')

    args = parser.parse_args()

    folder_path = r"D:\RumorDetection\train_results\train_mlp_rumor_train_BCE_2023-12-08_09-05-30\trained_model"

    files = list_files_in_directory(folder_path)

    father_folder_name, folder_name = make_father_folder(args)

    logger = make_logger(father_folder_name, folder_name)

    # test_ens(args, files, logger)

    output, id = test_ens(args, files, logger)

    df = pd.DataFrame()

    # df['id'] = id
    #
    # df['output_7'] = output.tolist()
    #
    # df.to_pickle(r'ens7.pkl')

    print()

    # for i in range(len(files)):
    #
    #     test(args, files[i],logger)


if __name__ == "__main__":
    main()
