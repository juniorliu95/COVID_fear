import os
import copy
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# select key events
def select_peaks(date_list, freq_list, lamb=1.8, pms=5, min_tweets=30):
    """
    function for selecting the peaks in the trendings
    return the list of index and value
    input:
        date_list: record of the date
        freq_list: record of the number of tweets
        lamb: the threshold of time of average used for judgement
        pms: the minimum difference of day between two peaks
        min_tweets: min number of tweets for a peak
    output:
        e_ind: index of the events
        e_val: value of the events
    """

    date_list = np.array(date_list)
    freq_list = np.array(freq_list)

    month_list = set([day[:-3] for day in date_list])
    month_list = list(sorted(month_list))
    e_ind, e_val = [], []

    for month in month_list:
        #generate the data within a month
        month_data = [] # record the value
        month_ind = [] # record the index
        month_day = [] # record the date
        for d_ind, day in enumerate(date_list):
            if len(month_data) > 31:
                break
            if month in day:
                month_day.append(day)
                month_data.append(freq_list[d_ind])
                month_ind.append(d_ind)

        month_data = np.array(month_data)
        month_day = np.array(month_day)
        # average per month as the baseline for selection
        ave = month_data.mean()

        for d_ind, day in enumerate(month_day):
            # according to Weber–Fechner law
            factor1 = month_data[d_ind]/ave > lamb
            # minimum number of tweets for a peak
            factor2 = month_data[d_ind] > min_tweets
            if factor1 and factor2:
                if len(e_ind) == 0:
                    print(day, month_data[d_ind])
                    e_ind.append(month_ind[d_ind])
                    e_val.append(month_data[d_ind])
                elif month_ind[d_ind] - e_ind[-1] >pms: # distance between two peak date 
                        print(day, month_data[d_ind])
                        e_ind.append(month_ind[d_ind])
                        e_val.append(month_data[d_ind])

    return e_ind, e_val


root_dir = os.getcwd()
data_path = os.path.join(root_dir, 'data')
result_path = os.path.join(root_dir, 'results', 'image')

# select event for the whole trend
df = pd.read_csv(os.path.join(data_path, 'daily_tweets.csv'))
date_list = df['Date'].to_list()
freq_list = df['Tweets'].to_list()
# import pdb;pdb.set_trace()
e_ind, e_val = select_peaks(date_list, freq_list, min_tweets=200)

# sort the events related to vaccination
v_val = [] # vector to record the number of tweets for events
v_ind = [] # index of events
for date in ['2021-02-14', '2021-02-17', '2021-04-13', '2021-05-21', '2021-05-24', '2021-06-21']:
    v_ind.append(date_list.index(date))
    v_val.append(freq_list[v_ind[-1]])

# plot the tweet trending
plt.figure(figsize=(20, 10))
plt.title('Tendency of vaccine-related tweets',fontsize=20)
plt.plot(np.arange(len(freq_list)), freq_list, 'b-')
plt.scatter(e_ind, e_val, s=100, c='r')
plt.scatter(v_ind, v_val, s=100, c='g')
plt.xticks(np.arange(len(freq_list), step=30), date_list[::30], fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Count of tweets',fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(result_path, 'num_of_tweets_with_events.png'))
plt.close()
print('Total saved...')


# select event for the sentiment trend
df = pd.read_csv(os.path.join(data_path, 'senti.csv'))
date_list = df['Date'].to_list()
pos_list = df['POSITIVE'].to_list()
neg_list = df['NEGATIVE'].to_list()
e_ind_pos, e_val_pos = select_peaks(date_list, pos_list, min_tweets=40)
print('positive done')
e_ind_neg, e_val_neg = select_peaks(date_list, neg_list, min_tweets=40)
print('negative done')
# plot the tweet trending
plt.figure(figsize=(20, 10))
plt.title('Tendency of vaccine-related tweets',fontsize=20)
# positive
plt.bar(np.arange(len(pos_list)), pos_list, width=1, color='chocolate', label='POSITIVE')
plt.scatter(e_ind_pos, e_val_pos, s=100, c='b')
# negative
plt.bar(np.arange(len(neg_list)), -1*np.array(neg_list), width=1, color='teal', label='NEGATIVE')
plt.scatter(e_ind_neg, -1*np.array(e_val_neg), s=100, c='r')

plt.yticks([-150, -100, -50, 0, 50, 100], [150, 100, 50, 0, 50, 100])
plt.xticks(np.arange(len(pos_list), step=30), date_list[::30], fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Count of tweets',fontsize=20)
plt.legend(loc=2, fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(result_path, 'num_of_tweets_with_senti.png'))
plt.close()
print('Senti saved...')