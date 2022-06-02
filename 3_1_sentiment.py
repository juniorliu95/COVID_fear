"""
3. statistical analysis on the sentiments of each day.
Make the Fig 4 in the manuscript
"""

#%%
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
matplotlib.rcParams['xtick.major.pad']='8'

title = "vaccine"
root_dir = os.getcwd()
result_root_dir = os.path.join(root_dir, 'results', 'image')
if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)

# dir of the texts to the sentiment results.
senti_dir = os.path.join(root_dir, 'results', 'sentiments_' + title)
data_dir = os.path.join(root_dir, 'data')

start_date = "2021-02-01"
end_date = "2021-09-30"
dates = pd.date_range(start_date, end_date, freq='D')
all_dates = dates.strftime('%Y-%m-%d').to_list()

pos_list = []
neg_list = []
neu_list = []
mix_list = []

if not os.path.exists(os.path.join(data_dir, title + '_senti.csv')):
    for d_ind, date in enumerate(all_dates):
        # judge if the date is within the selected date range
        if '-' not in date:
            continue
        print(date)
        pos = neg = neu = mix = 0
        for tweet_file in os.listdir(os.path.join(senti_dir, date)):
            with open(os.path.join(senti_dir, date, tweet_file), 'r') as f:
                line = f.readline()
                senti = line.split(',')[2]
                if senti == 'POSITIVE':
                    pos += 1
                elif senti == 'NEGATIVE':
                    neg += 1
                elif senti == 'NEUTRAL':
                    neu += 1 
                elif senti.upper() == 'MIXED':
                    mix += 1 
        assert pos+neg+neu+mix == len(os.listdir(os.path.join(senti_dir, date)))
        pos_list.append(pos)
        neg_list.append(neg)
        neu_list.append(neu)
        mix_list.append(mix)

    # save the results to a csv file
    df = pd.DataFrame(np.array([all_dates, pos_list, neg_list, neu_list, mix_list]).T, columns=['Date', 'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED'])

    df.to_csv(os.path.join(data_dir, title + '_senti.csv'), index=False)

else: # if the senti.csv is already saved.
    print('sentiments already saved. Using the csv file directly.....')
    df = pd.read_csv(os.path.join(data_dir, title + '_senti.csv'))
# import pdb;pdb.set_trace()

# get the index of each 1st and 6.30
mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_df_list = df.Date.to_list()
date_index_list = [date_df_list.index(date) for date in mark_date_list]

# the bar plot for sentiment
# Fig 4 in the paper
fig, ax = plt.subplots(figsize=(20, 10))
fig.subplots_adjust(right=0.75)
df_date = df['Date'].values
df_pos = df['POSITIVE'].values
df_neg = df['NEGATIVE'].values
plt.figure(figsize=(20, 10))
plt.bar(df_date, df_pos, color=(215/255., 131/255., 2/255.), width=1, label='Positive')
plt.bar(df_date, -1*df_neg, color=(18/255., 109/255., 131/255.), width=1, label='Negative')

if title == "vaccine":
    plt.yticks([-200, -150, -100, -50, 0, 50, 100, 150, 200], [200, 150, 100, 50, 0, 50, 100, 150, 200])
else:
    plt.yticks([-10, 0, 10], [10, 0, 10])
plt.xlim(0, len(df['Date']))
plt.xticks(date_index_list, mark_date_list, fontsize=20)
plt.setp(plt.yticks(fontsize=20))
plt.xlabel("Date", fontsize=20)
plt.ylabel("Count of tweets", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(result_root_dir, f'sentiment_{title}_bar_updown_2.png'))
plt.close()
