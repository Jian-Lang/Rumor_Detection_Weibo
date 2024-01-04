"""
@author: Lobster
@software: PyCharm
@file: test_tree.py
@time: 2023/12/19 14:47
"""
import pandas as pd

df = pd.read_pickle(r'D:\RumorDetection\data\common_data\reposts.pkl')
df.sort_values(by=['avg_length','max_length'],ascending=False,inplace=True)
print()