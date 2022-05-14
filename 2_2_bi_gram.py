"""Make the n-gram results of the japanese"""

import os
import re
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import ginza
# stop_words = list(ginza.STOP_WORDS)
from ginza import *

import spacy
nlp = spacy.load('ja_ginza')
# nlp.Defaults.stop_words # the stop words of japanese

from utils import remove_string_special_characters, remove_keywords

title = "vaccine"
# title = "vaccine_and_olympics"
root_dir = os.getcwd()
input_path = os.path.join(root_dir, 'results', title)
date_list = list(sorted(os.listdir(input_path)))
result_path = os.path.join(root_dir, 'data')

text_list = []
for date in date_list:
    print(date)
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
        doc = nlp(text)
        text = ' '.join([x.string for x in doc if not x.is_stop])
        text = remove_keywords(text)

        text_list.append(text)

vectorizer = CountVectorizer(ngram_range=(2,2)).fit(text_list) 
bag_of_words = vectorizer.transform(text_list)
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

with open(os.path.join(result_path, title + '_n_gram.pkl'),'wb') as f:
    pickle.dump(words_freq, f)