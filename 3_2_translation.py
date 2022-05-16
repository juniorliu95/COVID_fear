"""
translate the text to english
get ready for emotion detection
"""

import os
import re
import time
import pickle

import numpy as np
from googletrans import Translator # googletrans==4.0.0-rc1
from utils import remove_string_special_characters

translator = Translator()

title = "vaccine"
root_dir = os.getcwd()
input_path = os.path.join(root_dir, 'results', title)

start_date = "2021-02-01"
end_date = "2021-07-01"

date_list = list(sorted(os.listdir(input_path)))
s_ind = date_list.index(start_date)
e_ind = date_list.index(end_date)

result_path = os.path.join(root_dir, 'results', 'translation_' + title)
if not os.path.exists(result_path):
    os.makedirs(result_path)

for date in date_list[s_ind:e_ind+1]:
    print(date)
    date_path = os.path.join(input_path, date)
    file_list = os.listdir(date_path)
    
    # path for translation of each date
    date_output_path = os.path.join(result_path, date)
    if not os.path.exists(date_output_path):
        os.makedirs(date_output_path)
    
    for file_name in file_list:
        text = None
        with open(os.path.join(date_path, file_name), 'r') as f:
            text = f.readline()
            f.close()
        text = remove_string_special_characters(text)
        if not text:
            continue
        try:
            text_en = translator.translate(text).text
        except:
            time.sleep(500)
            text_en = translator.translate(text).text

        time.sleep(1)

        with open(os.path.join(date_output_path, file_name), "w") as f:
            f.write(text_en)
            f.close()
