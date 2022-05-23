"""
make the LDA of fear emotion
Fig 6, 7 in the manuscript
"""

#%%
import os
import re
import pickle
import time
import copy

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
matplotlib.rcParams['xtick.major.pad']='8'
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import ginza
from ginza import *
import spacy
nlp = spacy.load('ja_ginza')

from utils import remove_string_special_characters, remove_keywords, replace_words, plot_top_words, plot_word_clouds

n_samples = None # number of samples. None for all samples
n_features = None # number of featurs. None for max features
n_components = 2 # number of topics for clustering
n_top_words = 10 # number of words shown in the plot
n_top_words_wc = 100 # number of words shown in the plot

title = "vaccine"
emotion = "fear"
root_dir = os.getcwd()
input_path = os.path.join(root_dir, 'results', "emotions", emotion)
result_path = os.path.join(root_dir, 'results', 'image')

new_stop_words = ['で', 'けど', 'ませ', 'って', 'まし', 'てる', ' rt ', ' for ', ' of ']
korona_words = ['新型', '肺炎', 'コロナ','新型コロナ', '新型コロナウイルス', '新型コロナウィルス', 'ウイルス', 'ウィルス', 'コロ', 'covid', ' cov ', 'coronavirus', 'covid-19', 'vaccine', 'ワクチン', '接種']
new_stop_words.extend(korona_words)

#%%
# LDA model
# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_model, lda = None, None
text_list = []
text_date_list = []
if not os.path.exists(os.path.join(root_dir, "data", "LDA_fear_model.pkl")):
    if not os.path.exists(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_data.pkl')):
        date_list = list(sorted(os.listdir(input_path)))
        for date in date_list:
            print(date.split(".")[0])
            df = pd.read_csv(os.path.join(input_path, date))
            text_date_list.extend([date.split(".")[0]]*len(df))
            for text in df["Text"].to_list():
                text = remove_string_special_characters(text)
                if text is None:
                    continue
                doc = nlp(text)
                text = ' '.join([x.string for x in doc if not x.is_stop])
                text = remove_keywords(text, additional=new_stop_words)
                text_list.append(text)

        with open(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_data.pkl'), 'wb') as f:
            pickle.dump({"Text":text_list, "Date":text_date_list}, f)
            print('data saved ...')
            f.close()
    else:
        with open(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_data.pkl'), 'rb') as f:
            dic_temp = pickle.load(f)
            text_list = dic_temp["Text"]
            text_date_list = dic_temp["Date"]
            print('data loaded ...')
            f.close()

    # get the LDA model for different sentiments
    if n_samples is not None:
        data_samples = text_list[:n_samples]
    else:
        n_samples = len(text_list)
        data_samples = text_list
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    t0 = time.time()
    tf_model = tf_vectorizer.fit(data_samples)
    tf = tf_model.transform(data_samples)
    print("done in %0.3fs." % (time.time() - t0))
    print()

    # Fit the LDA model
    if n_features is not None:
        print('\n' * 2, "Fitting LDA models with tf fe`atures, "
        "n_samples=%d and n_features=%d..."
        % (n_samples, n_features))
    else:
        print('\n' * 2, "Fitting LDA models with tf features, "
        "n_samples=%d and n_features=max..."
        % (n_samples))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=5)
    t0 = time.time()
    lda.fit(tf)
    print("done in %0.3fs." % (time.time() - t0))
    
    tf_feature_names = tf_vectorizer.get_feature_names()
    tf_feature_names_temp = copy.deepcopy(tf_feature_names)
    for t_ind, tf_feature_name in enumerate(tf_feature_names_temp):
        tf_feature_name = replace_words(tf_feature_name)
        tf_feature_names[t_ind] = tf_feature_name
    del tf_feature_names_temp

    with open(os.path.join(root_dir, "data", "LDA_fear_model.pkl"), "wb") as f:
        pickle.dump({'model':lda, 'name':tf_feature_names}, f)
        f.close()
    with open(os.path.join(root_dir, "data", "LDA_fear_tf_model.pkl"), "wb") as f:
        pickle.dump(tf_model, f)
        f.close()
else:
    with open(os.path.join(root_dir, "data", "LDA_fear_model.pkl"), "rb") as f:
        temp = pickle.load(f)
        lda = temp['model']
        tf_feature_names = temp['name']
        f.close()
    with open(os.path.join(root_dir, "data", "LDA_fear_tf_model.pkl"), "rb") as f:
        tf_model = pickle.load(f)
        f.close()
    print("model loaded successfully!")
#%%
# translation
trans_path = os.path.join(root_dir, "data", "LDA_fear_trans.csv")
df_trans = None
if os.path.exists(trans_path):
    df_trans = pd.read_csv(trans_path)
    df_trans_temp = df_trans.iloc[:,3:]
    df_trans = df_trans.iloc[:,1:3]
    df_trans.rename(columns={"topic 1 jp": "jp", "topic 1 en": "en"}, inplace=True)
    df_trans_temp.rename(columns={"topic 2 jp": "jp", "topic 2 en": "en"}, inplace=True)
    df_trans = df_trans.append(df_trans_temp)
    del df_trans_temp


#%%
# Fig 6
plot_top_words(lda, tf_feature_names, n_top_words, '', title + "_" + emotion, n_components, figsize=(15, 10))
plot_word_clouds(lda, tf_feature_names, n_top_words_wc, f'Topics in {title} LDA model of {emotion} emotion', title + "_" + emotion, n_components)

# english version
plot_top_words(lda, tf_feature_names, n_top_words, '', title + "_" + emotion, n_components, en=True, trans_name=f'LDA_{emotion}_trans.csv', trans=df_trans, figsize=(15, 10))
plot_word_clouds(lda, tf_feature_names, n_top_words_wc, f'Topics in {title} LDA model of {emotion} emotion', title + "_" + emotion, n_components, en=True, trans=df_trans)

#%%
# # save the fear texts
# import pandas as pd
# df = pd.DataFrame(text_list, columns=['text'])
# df.to_csv(os.path.join(root_dir, 'results', 'data', f'LDA_{emotion}_text.csv'), index=False)


#%%
# get the source of each document
start_date = "2021-02-01"
end_date = "2021-09-30"
dates = pd.date_range(start_date, end_date, freq='D')
dates = dates.strftime('%Y-%m-%d').to_list()

mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_index_list = [dates.index(date) for date in mark_date_list]

df_topic = None
if not os.path.exists(os.path.join(root_dir, "data", "LDA_fear_topic.pkl")):
    df_text = pd.DataFrame(list(zip(text_date_list, text_list)), columns=['Date', "Text"])
    topic_list = []
    for date in dates:
        text_list_day = df_text.loc[df_text.Date==date]["Text"].to_list()
        tf_day = tf_model.transform(text_list_day)
        lda_day = lda.transform(tf_day)
        t1 = lda_day[:,0].sum()
        t2 = lda_day[:,1].sum()
        topic_list.append([t1, t2])
    df_topic = pd.DataFrame(topic_list, columns=["Infection", "Vaccine confidence"])
    with open(os.path.join(root_dir, "data", "LDA_fear_topic.pkl"), "wb") as f:
        pickle.dump(df_topic, f)
        f.close()
else:
    with open(os.path.join(root_dir, "data", "LDA_fear_topic.pkl"), "rb") as f:
        df_topic = pickle.load(f)
        f.close()

#%%
# check if ratio > 1 by t-test
from scipy import stats
result = stats.ttest_1samp((df_topic["Infection"]/df_topic["Vaccine confidence"]).to_list(), 1)
result

# %%
# Fig 7
# draw the trend of different topics
# sb.lineplot(data=df_topic)
plt.figure(figsize=(10, 5))
plt.plot(df_topic["Infection"]/df_topic["Vaccine confidence"])
plt.axhline(1, ls="--", c="k")
plt.xticks(date_index_list, mark_date_list)
plt.xlim(0, len(dates))
plt.yscale("log")
plt.ylim(0.4, 2.5)
plt.ylabel('Infection/Vaccine confidence (log)')
plt.xlabel("Date")
plt.savefig(os.path.join(root_dir, "results", "image", "emotion_fear_ratio.png"))

# %%
