import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from evaluation.environment import TrainingEnvironment, TestingEnvironment
from concurrent.futures import ThreadPoolExecutor

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 目前，跨多個GPU的內存增長需求必須相同
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 選擇第1個GPU
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # GPU初始化前必須設定內存增長
        print(e)


def run_simulation(train_env, output_file_path):
    clicked_ids_collection = []  # 獨立的列表，每個環境一個

    for epoch in range(1, 1001):
        train_env.reset()

        while train_env.has_next_state():
            cur_user = train_env.get_state()
            slate = random.sample(range(209527), 5)
            clicked_id, in_environment = train_env.get_response(slate)

            if clicked_id != -1:
                clicked_ids_collection.append((cur_user, clicked_id))

        train_score = train_env.get_score()
        df_train_score = pd.DataFrame([[user_id, score] for user_id, score in enumerate(train_score)], columns=['user_id', 'avg_score'])
        df_train_score

        # 每過10個epoch就將該環境的數據寫入文件
        if epoch % 10 == 0:
            with open(output_file_path, 'a') as f:
                for user_id, item_id in clicked_ids_collection:
                    f.write(f"{user_id}, {item_id}\n")
            clicked_ids_collection = []  # 重置該環境的列表


# 初始化訓練環境
train_env = TrainingEnvironment()

# 平行執行的環境數量
num_environments = os.cpu_count()  # 使用可用的CPU核心數

# 使用ThreadPoolExecutor進行平行執行
with ThreadPoolExecutor(max_workers=num_environments) as executor:
    # 存儲future的列表
    futures = []

    # 定義文件路徑
    output_file_path = 'clicked_ids_output_final.txt'

    # 提交每個環境給執行器
    for _ in range(num_environments):
        future = executor.submit(run_simulation, TrainingEnvironment(), output_file_path)
        futures.append(future)

    # 等待所有future完成
    for future in tqdm(futures, desc="模擬", unit="env"):
        future.result()
