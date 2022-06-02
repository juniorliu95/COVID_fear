"""
Select the best number of topics for LDA modelling.
Fig 1 in the appendix
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
from sklearn.model_selection import GridSearchCV

import ginza
from ginza import *
import spacy
nlp = spacy.load('ja_ginza')

from utils import remove_string_special_characters, remove_keywords

n_samples = None # number of samples. None for all samples
n_features = None # number of features. None for max features

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

# %%
model = None # define the LDA cross validation model
if not os.path.exists(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_CV_model.pkl')):
    text_list = []
    text_date_list = []

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

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english') # dealt with Japanese stop words in the last step
    t0 = time.time()
    tf_model = tf_vectorizer.fit(data_samples)
    tf = tf_model.transform(data_samples)
    print("done in %0.3fs." % (time.time() - t0))

    # LDA model
    # Define Search Param
    n_topics = [2, 4, 6, 8, 10, 20, 30, 40, 50]
    # n_topics = [2, 4,]
    search_params = {'n_components': n_topics}

    # Init the Model
    lda = LatentDirichletAllocation(max_iter=5, learning_method='online',learning_offset=50.,random_state=5)

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(tf)
    with open(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_CV_model.pkl'), "wb") as f:
        pickle.dump(model, f)
        f.close()
else:
    with open(os.path.join(root_dir, 'data', f'{title}_LDA_{emotion}_CV_model.pkl'), "rb") as f:
        model = pickle.load(f)
        f.close()

#%%
# Get Log Likelyhoods from Grid Search Output
# Fig 1 in the appendix
log_likelihoods = model.cv_results_['mean_test_score']
print("best number of topics: ", model.best_params_)

# Show graph
n_topics = [2, 4, 6, 8, 10, 20, 30, 40, 50]
plt.plot(n_topics, log_likelihoods, color=(18/255., 104/255., 131/255.))
plt.xticks(n_topics)
# plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel(" Mean log likelihood scores")
plt.savefig(os.path.join(result_path, "LDA_fear_topic_num.png"))

# %%
