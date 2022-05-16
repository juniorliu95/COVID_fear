"""2. clean the dataset by keywords"""

import os
import shutil
import codecs

title = 'vaccine'

root_dir = os.getcwd()
result_root_dir = os.path.join(root_dir, 'results', title)
if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)

# dir of the texts to be cleaned.
start_date = '2021-02-01'
end_date = '2021-10-01'
data_dir = os.path.join(root_dir, 'results', 'text')
all_date_list = list(sorted(os.listdir(data_dir)))

if start_date:
    all_date_list_temp = []
    record = False
    for date in all_date_list:
        if date == start_date:
            record = True
        if date == end_date:
            record = False
        if record:
            all_date_list_temp.append(date)
    all_date_list = all_date_list_temp

keywords = ['ワクチン','接種','せっしゅ','注射','ちゅうしゃ','打つ', '投与', 'とうよ','mRNA', '副作用', 'ふくさよう', '副反応','ふくはんのう', 'vaccine','vacc', 'pfizer','ファイザー', 'astra','zeneca','アストラ','ゼネカ', 'modelna','moderna', 'モデルナ']

for d_ind, date in enumerate(all_date_list):
    # judge if the date is within the selected date range
    if '-' not in date:
        continue
    print(date)
    # make the path of the result
    if not os.path.exists(os.path.join(result_root_dir, date)):
        os.makedirs(os.path.join(result_root_dir, date))

    for tweet_file in os.listdir(os.path.join(data_dir, date)):
        with codecs.open(os.path.join(data_dir, date, tweet_file), 'r', 'utf-8') as f:
            text = f.readline().strip()
            if "まとまっ" in text and "スマート" in text and "ニュース" in text: # reject advertisements
                continue
            
            for keyword in keywords:
                if keyword in text.lower():
                    shutil.copyfile(os.path.join(data_dir, date, tweet_file), os.path.join(result_root_dir, date, tweet_file))
                    break
            f.close()
        

