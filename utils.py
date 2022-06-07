from logging import root
import os
import re
import unicodedata
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import ginza
from ginza import *
import spacy
nlp = spacy.load('ja_ginza')
# nlp.Defaults.stop_words # the stop words of japanese
from wordcloud import WordCloud
from googletrans import Translator # googletrans==4.0.0-rc1

def remove_string_special_characters(s): 
    """
    remove some special characters and other components from the string.
    Input:
        s: The string to be cleaned.
        additional: if not None, a list of additional rules to be cleaned.
    """
    # Halfwidth form <- fullwidth forms
    stripped = unicodedata.normalize("NFKC", s)
    
    # Change any white space to one space 
    stripped = re.sub('\s+', ' ', stripped) 
    
    # Remove urls
    stripped = re.sub(r"http\S+",'', stripped)
    
    # remove including after @ marks
    stripped = re.sub('\s@\S+\s', ' ', stripped)

    # remove the punctuations
    stripped = re.sub('[,.;。、；]', ' ', stripped)

    # removes special characters with '' 
    stripped = re.sub('[^一-龠ぁ-ゔァ-ヴーa-zA-Z0-9ａ-ｚＡ-Ｚ０-９\s]', '', stripped)
    stripped = re.sub('_', '', stripped)
    
    # lower case
    stripped = stripped.lower()

    # remove amp
    stripped = stripped.replace(' amp ', '') 
    # Remove numbers
    stripped = re.sub(r"[0-9０-９]+",'', stripped)

    # Remove start and end white spaces 
    stripped = stripped.strip()
    stripped = " ".join(stripped.split())
    if stripped != '': 
        return stripped

def remove_keywords(s, additional=None):
    stripped = " " + s + " "
    # Remove keywords
    new_stop_words = ['で', 'けど', 'ませ', 'って', 'まし', 'てる', 'だろう', 'しよう', 'しょう', 'しょ', 'じゃ', 'rt', 'for', 'of', 'to', 'the', 'in', 'gt']
    for new_stop_word in new_stop_words:
        stripped = re.sub("\s"+new_stop_word+"\s",' ', stripped)

    # change words
    if additional is not None:
        for add in additional:
            stripped = re.sub(add, ' ', stripped)
    
    stripped = stripped.strip()
    # remove white spaces
    stripped = " ".join(stripped.split())

    return stripped

def replace_words(word):
    """replace some wrong spells in the words"""
    # list of words need all capitalize
    capital_list = ['cdc', 'nhk', 'eu', 'fda', 'who']
    # list of words need first letter capitalize
    first_capital_list = ['japan', 'reuters', 'yahoo']

    if word in capital_list:
        return word.upper()
    elif word in first_capital_list:
        return word.capitalize()
    elif word == 'mrna':
        return 'mRNA'
    return word

def load_correct_trans(file_path, sheet_name):
    """load correct translation"""
    root_dir = os.getcwd()
    data = pd.read_excel(os.path.join(root_dir, file_path), sheet_name)
    return data


# LDA make top word lists
def plot_top_words(model, feature_names, n_top_words, title, save_name=None, n_components=3, root_dir='./', en=False, trans_name='LDA_trans.csv', trans=None, figsize=(30, 15), columns=1):
    fig, axes = plt.subplots(columns, np.ceil(n_components/columns).astype(int), figsize=figsize, sharex=True)
    axes = axes.flatten()

    top_features_trans = []

    for topic_idx, topic in enumerate(model.components_[::-1]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]

        if en: # if make the english version
            translator = Translator()
            top_features_temp = []
            for top_feature in top_features:
                if trans is not None and len(trans.loc[trans.jp==top_feature]['en']) != 0:
                    top_feature_en = trans.loc[trans.jp==top_feature]['en'].values[0]
                else:
                    time.sleep(1)
                    top_feature_en = translator.translate(top_feature).text
                top_features_temp.append(top_feature_en)
            top_features_trans.append(copy.deepcopy(top_features))
            top_features_trans.append(copy.deepcopy(top_features_temp))

            top_features = top_features_temp

        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color=(18/255., 104/255., 131/255.))
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
        plt.tight_layout()
    
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    name = 'LDA'
    if save_name is not None:
        name = save_name + '_' + name
    if en:
        plt.savefig(os.path.join(root_dir, 'results', 'image', name+'_en.png'), dpi=300)
    else:
        plt.savefig(os.path.join(root_dir, 'results', 'image', name+'.png'), dpi=300)

    # save the translations
    if en:
        columns = []
        for i in range(len(top_features_trans)//2):
           columns.append(f'topic {i+1} jp')
           columns.append(f'topic {i+1} en')

        # make a translation list
        df = pd.DataFrame(data=list(zip(*top_features_trans)), columns=columns, index=None)
        df.to_csv(os.path.join(root_dir, 'data', trans_name))

# LDA make word cloud
def plot_word_clouds(model, feature_names, n_top_words, title, save_name=None, n_components=3, root_dir='./', en=False, trans=None, figsize=(30, 15), columns=1):
    fig, axes = plt.subplots(columns, np.ceil(n_components / columns).astype(int), figsize=figsize, sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_[::-1]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]

        if en: # if make the english version
            translator = Translator()
            top_features_temp = []
            for top_feature in top_features:
                if (trans is not None) and len(trans.loc[trans.jp==top_feature])!= 0:
                    top_feature_en = trans.loc[trans.jp==top_feature]['en'].values[0]
                else:
                    time.sleep(1)
                    top_feature_en = translator.translate(top_feature).text
                top_features_temp.append(top_feature_en)
            top_features = top_features_temp
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})

        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)
        font_path = os.path.join(root_dir, 'font', 'Boku2-Bold.otf')
        wc = WordCloud(font_path=font_path, background_color="white", max_words=n_top_words, mask=mask)
        # generate word cloud
        freqencies = dict(zip(top_features, weights))
        wc.generate_from_frequencies(freqencies)

        # show
        ax.imshow(wc, interpolation="bilinear")
        ax.tick_params(axis='both', which='major', labelsize=20)
        # for i in 'top right left'.split():
        #     ax.spines[i].set_visible(False)
        ax.axis("off")
        fig.suptitle(title, fontsize=40)
        plt.tight_layout()

    name = 'LDA'
    if save_name is not None:
        name = save_name+'_'+name

    # plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    if en:
        plt.savefig(os.path.join(root_dir, 'results', 'image', name+'_wc_en.png'), dpi=300)
    else:
        plt.savefig(os.path.join(root_dir, 'results', 'image', name+'_wc.png'), dpi=300)


if __name__ == '__main__':
    df = load_correct_trans("data/Appendix 2_Translation Table.xlsx", "1_Data cleaning keywords")
    import pdb;pdb.set_trace()
    pass