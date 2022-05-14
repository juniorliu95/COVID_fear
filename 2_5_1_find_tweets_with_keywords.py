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
# stop_words = list(ginza.STOP_WORDS)
from ginza import *
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm
from wordcloud import WordCloud
from googletrans import Translator # googletrans==4.0.0-rc1

import spacy
nlp = spacy.load('ja_ginza')
# nlp.Defaults.stop_words # the stop words of japanese

from utils import remove_string_special_characters, remove_keywords

translator = Translator()

title = "vaccine"
# title = "vaccine_and_olympics"
root_dir = os.getcwd()
input_path = os.path.join(root_dir, 'results', title)
date_list = list(sorted(os.listdir(input_path)))

keywords = [
    "感染", 
    # "予約", 
    "会場", 
    # "情報",
    # "日本",
    # "副反応",
    # "ファイザー",
    # "可能",
    # "効果",
    # "デルタ",
    # "以上",
    # "感染者",
    # "モデルナ",
    # "変異", 
]
keywords_title = "+".join(keywords)
print(keywords_title)
result_path = os.path.join(root_dir, 'results', 'keywords', keywords_title)
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
        #! add keywords judgement
        mark = True
        for keyword in keywords:
            if keyword not in text:
                mark = False
        if mark:
            text_list.append([date, file_name, text])

df_keywords = pd.DataFrame(text_list, columns=["Date", "ID", "Text"])
df_keywords.to_csv(os.path.join(result_path, f"{keywords_title}.csv"))
del text_list, df_keywords

print("Done!")

# %%


# %%
