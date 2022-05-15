"""
get the tweets of the specified keywords
and make csv files
"""
#%%
import os
import re
import pickle
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import ginza
from ginza import *
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm
from wordcloud import WordCloud
from googletrans import Translator # googletrans==4.0.0-rc1

import spacy
nlp = spacy.load('ja_ginza')

from utils import remove_string_special_characters, remove_keywords

translator = Translator()

title = "vaccine"
# title = "vaccine_and_olympics"
root_dir = os.getcwd()
input_path = os.path.join(root_dir, 'results', title)
date_list = list(sorted(os.listdir(input_path)))

keywords = [
    # 'アストラ ゼネカ', 
    # '予約 可能', 
    # 'article reuters', 
    # '会場 予約', 
    '医療 従事者',
]
keyword = keywords[0]
print(keyword)
result_path = os.path.join(root_dir, 'results', 'keywords', keyword)
if not os.path.exists(result_path):
    os.makedirs(result_path)

start, end = '2021-02-01', '2021-09-30'
print(f"{start}~{end}")

#%%
start = date_list.index(start)
end = date_list.index(end)
period_date_list = date_list[start:end+1]

text_list = []
for date in tqdm(period_date_list):
    date_path = os.path.join(input_path, date)
    file_list = os.listdir(date_path)
    
    for file_name in file_list:
        text = None
        with open(os.path.join(date_path, file_name), 'r') as f:
            text = f.readline()
            f.close()

        pre_text = remove_string_special_characters(text)
        if pre_text is None:
            continue
        doc = nlp(pre_text)
        pre_text = ' '.join([x.string for x in doc if not x.is_stop])
        pre_text = remove_keywords(pre_text)

        if keyword in pre_text:
            text_list.append([date, file_name, text])

df_keywords = pd.DataFrame(text_list, columns=["Date", "ID", "Text"])
df_keywords.to_csv(os.path.join(result_path, f"{keyword}.csv"))
del text_list, df_keywords

print("Done!")
