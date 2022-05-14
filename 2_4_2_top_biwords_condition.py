"""
Make the 2-gram results of the japanese
Save the results of different time periods
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
result_path = os.path.join(root_dir, 'results', 'image')

save_name = "stages_bi"

month_list = ["February", "March", "April", "May", "June", "July", "August", "September"]
state_list = ["{:02d}".format(i) for i in range(2, 10)]
state_list.append("total")

# kick out vaccine-related words
new_stop_words = ['で', 'けど', 'ませ', 'って', 'まし', 'てる', ' rt ', ' for ', ' of ']
korona_words = ['新型', '肺炎', 'コロナ','新型コロナ', '新型コロナウイルス', '新型コロナウィルス', 'ウイルス', 'ウィルス', 'コロ', 'covid', ' cov ', 'coronavirus', 'covid-19', 'vaccine', 'ワクチン', '接種']
new_stop_words.extend(korona_words)

#%%
# plot the top words
def plot_top_words(top_feat, weights, topic_keys, topic_names, title, save_name, result_path, top_num=50, n_components=3, en=False, trans=None):
    """
    top_features: dict, each is the top keywords of a vaccine_name
    weights: dict, each is the weight of top keywords of a vaccine_name
    """
    fig, axes = plt.subplots(2, n_components//2, figsize=(30, 20), sharex=True)
    axes = axes.flatten()

    top_features = deepcopy(top_feat)

    top_features_trans = []

    for topic_idx, topic_key in enumerate(topic_keys):
        ax = axes[topic_idx]
        top_features[topic_key] = top_features[topic_key][:top_num]
        weights[topic_key] = weights[topic_key][:top_num]
        if en:
            top_features_temp = []
            for top_feature in tqdm(top_features[topic_key],'translation'):
                top_feature_en = None
                if trans is not None and len(trans.loc[trans.jp==top_feature]['en']) != 0:
                    top_feature_en = trans.loc[trans.jp==top_feature]['en'].values[0]
                else:
                    time.sleep(1)
                    top_feature_en = translator.translate(top_feature).text
                top_features_temp.append(top_feature_en)
            top_features_trans.append(deepcopy(top_features[topic_key]))
            top_features_trans.append(deepcopy(top_features_temp))

            top_features[topic_key] = top_features_temp

        ax.barh(top_features[topic_key], weights[topic_key], height=0.7, alpha=0.5)
        ax.set_title(f'{topic_names[topic_idx]}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
        plt.tight_layout()

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    if en:
        plt.savefig(os.path.join(result_path, '{}_topwords_en.png'.format(save_name)))
    else:
        plt.savefig(os.path.join(result_path, '{}_topwords.png'.format(save_name)))

    # save the translations
    if en:
        columns = []
        for i in range(len(top_features_trans)//2):
           columns.append(f'{topic_keys[i]} jp')
           columns.append(f'{topic_keys[i]} en')

        # make a translation list
        df = pd.DataFrame(data=list(zip(*top_features_trans)), columns=columns, index=None)
        df.to_csv(os.path.join(root_dir, 'data', 'top_biwords_conditions_trans.csv'))

#%%
def plot_word_clouds(top_feat, weights, topic_keys, topic_names, title, save_name, result_path, top_num=50, n_components=3, en=False, trans=None):
    """
    top_features: dict, each is the top keywords of a vaccine_name
    weights: dict, , each is the weight of top keywords of a vaccine_name
    """

    fig, axes = plt.subplots(2, n_components//2, figsize=(30, 20), sharex=True)
    axes = axes.flatten()
    top_features = deepcopy(top_feat)

    for topic_idx, topic_key in enumerate(topic_keys):
        ax = axes[topic_idx]
        top_features[topic_key] = top_features[topic_key][:top_num]
        weights[topic_key] = weights[topic_key][:top_num]
        if en:
            top_feature_temp = []
            for top_feature in tqdm(top_features[topic_key],'translation'):
                top_feature_en = None
                if trans is not None and len(trans.loc[trans.jp==top_feature]['en']) != 0:
                    top_feature_en = trans.loc[trans.jp==top_feature]['en'].values[0]
                else:
                    time.sleep(1)
                    top_feature_en = translator.translate(top_feature).text
                top_feature_temp.append(top_feature_en)
            top_features[topic_key] = top_feature_temp

        ax.set_title(f'{topic_names[topic_idx]}',
                     fontdict={'fontsize': 30})

        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)
        font_path = os.path.join(root_dir, 'font', 'Boku2-Bold.otf')
        wc = WordCloud(font_path=font_path, background_color="white", max_words=len(weights[topic_key]), mask=mask)
        # generate word cloud
        freqencies = dict(zip(top_features[topic_key], weights[topic_key]))
        wc.generate_from_frequencies(freqencies)

        # show
        ax.imshow(wc, interpolation="bilinear")
        ax.tick_params(axis='both', which='major', labelsize=20)
        # for i in 'top right left'.split():
        #     ax.spines[i].set_visible(False)
        ax.axis("off")
        fig.suptitle(title, fontsize=40)
        plt.tight_layout()

    # plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    if en:
        plt.savefig(os.path.join(result_path, '{}_topwords_wc_en.png'.format(save_name)))
    else:
        plt.savefig(os.path.join(result_path, '{}_topwords_wc.png'.format(save_name)))

#%%
top_feat = dict()
weights = dict()

if not os.path.exists(os.path.join(result_path, title + '_top_biwords_condition.pkl')):
    for i in range(len(state_list)):
        month = state_list[i] # format 02d
        period_date_list = date_list

        if month != "total":
            start = date_list.index(f"2021-{month}-01")
            end = len(date_list)
            if i != len(state_list) - 2:
                end = date_list.index(f"2021-{state_list[i + 1]}-01")
            period_date_list = period_date_list[start:end]
        else:
            start = date_list.index("2021-02-01")
            period_date_list = period_date_list[start:]

        text_list = []
        for date in tqdm(period_date_list):
            if month != "total" and date.split("-")[1] != month:
                break
            date_path = os.path.join(input_path, date)
            file_list = os.listdir(date_path)
            
            for file_name in file_list:
                text = None
                with open(os.path.join(date_path, file_name), 'r') as f:
                    text = f.readline()
                    f.close()
                text = remove_string_special_characters(text)
                if text is None:
                    continue
                if "まとまっ" in text and "スマート" in text and "ニュース" in text:
                    continue
                doc = nlp(text)
                text = ' '.join([x.string for x in doc if not x.is_stop])
                text = remove_keywords(text, additional=new_stop_words)
                text_list.append(text)

        vectorizer = CountVectorizer(ngram_range=(2, 2)).fit(text_list)
        bag_of_words = vectorizer.transform(text_list)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

        top_feat[state_list[i]] = [w[0] for w in words_freq]
        weights[state_list[i]] = [w[1]/len(period_date_list) for w in words_freq]

    with open(os.path.join(result_path, title + '_top_biwords_condition.pkl'),'wb') as f:
        pickle.dump([top_feat, weights], f)
        print("saved...")
        f.close()
else:
    with open(os.path.join(result_path, title + '_top_biwords_condition.pkl'),'rb') as f:
        [top_feat, weights] = pickle.load(f)
        print("read...")
        f.close()

#%%
# trend of top words
import ruptures as rpt
top_1000 = np.array(weights["total"][:1000])
algo = rpt.Pelt(model="rbf").fit(top_1000)
result = algo.predict(pen=1)

# display
rpt.display(top_1000, result)
plt.show()

#%%
## ['アストラ ぜねか', '予約 可能', 'article Reuters', '会場 予約', '医療 従業者']
plt.figure(figsize=(10, 5))
plt.bar(list(range(1, 11)), weights["total"][:10], alpha=0.5)
plt.plot(list(range(1, 11)), weights["total"][:10], c="r", marker=".")
label_trans = ["Astra\nZeneca", "reservation\npossible", "article\nReuters", "venue\nreserve", "medical care\nworkers", "venue\nTokyo", "possible\nreservation", "chat\nwatch", "infection\nexpansion", "Tokyo\nOpympics"]
plt.xticks(list(range(1, 11)), label_trans, rotation=45)
# plt.close()

#%%
# statistics
# top_feat dict of months and frequency of top words_freq
# state_list: months
trend_total = [sum(weights[month]) for month in state_list]

from scipy.stats import linregress, levene
result = linregress(np.arange(len(trend_total)), trend_total)

## levene variance
# print(levene(*[weights[month] for month in state_list]))

# weights


#%%
df_trans = None
if os.path.exists(os.path.join(root_dir, 'data', 'top_biwords_conditions_trans.csv')):
    df_trans = pd.read_csv(os.path.join(root_dir, 'data', 'top_biwords_conditions_trans.csv'))

    df_trans_sort = df_trans.iloc[:,1:3]
    df_trans_sort = df_trans_sort.rename(columns={'02 jp':'jp', '02 en':'en'})
    for i in range(1, len(state_list)):
        df_trans_1 = df_trans.iloc[:,i*2+1:i*2+3]
        df_trans_1 = df_trans_1.rename(columns={f'{state_list[i]} jp':'jp', f'{state_list[i]} en':'en'})
        df_trans_sort = df_trans_sort.append(df_trans_1)
        del df_trans_1

    df_trans = df_trans_sort.drop_duplicates(subset=['jp'])
    del df_trans_sort

#%%
with open(os.path.join(result_path, title + '_top_biwords_condition.pkl'), 'rb') as f:
    top_feat, weights = pickle.load(f)
    f.close()

state_list = ["{:02d}".format(i) for i in range(2, 10)]

top_num = 10 # number of top words
plot_top_words(top_feat, weights, state_list, month_list, title, save_name, result_path, top_num=top_num, n_components=len(state_list), en=False, trans=None)
plot_top_words(top_feat, weights, state_list, month_list, title, save_name, result_path, top_num=top_num, n_components=len(state_list), en=True, trans=df_trans)
plot_word_clouds(top_feat, weights, state_list, month_list, title, save_name, result_path, top_num=top_num, n_components=len(state_list), en=False, trans=None)
plot_word_clouds(top_feat, weights, state_list, month_list, title, save_name, result_path, top_num=top_num, n_components=len(state_list), en=True, trans=df_trans)

# %%

