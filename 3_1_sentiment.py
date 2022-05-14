"""3. statistical analysis on the snetiments of each day"""
#%%
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
matplotlib.rcParams['xtick.major.pad']='8'

title = "vaccine"
# title = "vaccine_and_olympics"
root_dir = os.getcwd()
result_root_dir = os.path.join(root_dir, 'results', 'image')
if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)

# dir of the texts to the sentiment results.
senti_dir = os.path.join(root_dir, 'results', 'sentiments_' + title)
data_dir = os.path.join(root_dir, 'data')
all_date_list = list(sorted(os.listdir(senti_dir)))
start_date = "2021-02-01"
end_date = "2021-09-30"
start, end = all_date_list.index(start_date), all_date_list.index(end_date)
all_dates= all_date_list[start:end+1]

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
import pdb;pdb.set_trace()

# get the index of each 1st and 6.30
mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_df_list = df.Date.to_list()
date_index_list = [date_df_list.index(date) for date in mark_date_list]

# total = df["POSITIVE"].sum() + df["NEGATIVE"].sum() + df["NEUTRAL"].sum() + df["MIXED"].sum()
# print("POSITIVE:", df["POSITIVE"].sum(), df["POSITIVE"].sum()/total)
# print("NEGATIVE:", df["NEGATIVE"].sum(), df["NEGATIVE"].sum()/total)
# print("NEUTRAL:", df["NEUTRAL"].sum(), df["NEUTRAL"].sum()/total)
# print("MIXED:", df["MIXED"].sum(), df["MIXED"].sum()/total)

# # draw the image
# # curve plot
# plt.figure(figsize=(20, 10))
# plt.title('Sentiments',fontsize=20)
# plt.plot(np.arange(len(df['Date'])), df['POSITIVE']+df['NEGATIVE']+df['NEUTRAL']+df['MIXED'], 'k', label='total')
# plt.plot(np.arange(len(df['POSITIVE'])), df['POSITIVE'], c='chocolate', label='positive')
# plt.plot(np.arange(len(df['NEGATIVE'])), df['NEGATIVE'], c='teal', label='negative')
# plt.plot(np.arange(len(df['NEUTRAL'])), df['NEUTRAL'], c='goldenrod', label='neutral')
# plt.plot(np.arange(len(df['MIXED'])), df['MIXED'], c='darkseagreen', label='mixed')

# plt.legend(fontsize=20)
# plt.xlabel('Date',fontsize=20)
# plt.ylabel('Count of tweets',fontsize=20)
# plt.xticks(date_index_list, mark_date_list, fontsize=20)
# plt.xlim(0, len(df['Date']))
# plt.setp(plt.yticks(fontsize=20))
# plt.tight_layout()
# plt.savefig(os.path.join(result_root_dir, f'sentiment_{title}.png'))
# plt.close()

# # curve pos/neg plot
# plt.figure(figsize=(20, 10))
# plt.title('Sentiments',fontsize=20)
# # fit a straight line
# x = np.arange(len(df['POSITIVE']))
# y = df['POSITIVE'].to_numpy()/df['NEGATIVE'].to_numpy()
# plt.plot(x, y)
# plt.xlabel('Date',fontsize=20)
# plt.ylabel('Count of tweets',fontsize=20)
# plt.ylim(0, 1)
# plt.xlim(0, len(df['Date']))
# plt.xticks(date_index_list, mark_date_list, fontsize=20)
# plt.setp(plt.yticks(fontsize=20))
# plt.tight_layout()
# plt.savefig(os.path.join(result_root_dir, f'sentiment_ratio_{title}.png'))
# plt.close()


# the bar plot for sentiment
fig, ax = plt.subplots(figsize=(20, 10))
fig.subplots_adjust(right=0.75)
df_date = df['Date'].values
df_pos = df['POSITIVE'].values
df_neg = df['NEGATIVE'].values
plt.figure(figsize=(20, 10))
plt.bar(df_date, df_pos, color='red', alpha=0.5, width=1, label='Positive')
plt.bar(df_date, -1*df_neg, alpha=0.5, width=1, label='Negative')

# # smoothed curve
# from scipy.signal import savgol_filter
# y_smooth1 = savgol_filter(-1*df_neg, 21, 3, mode= 'nearest')
# plt.plot(df_date, y_smooth1, "b-")
# y_smooth2 = savgol_filter(df_pos, 21, 3, mode= 'nearest')
# plt.plot(df_date, y_smooth2, "r-")

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

# # image with daily data
# df_death = pd.read_csv(os.path.join(data_dir, 'death_daily.csv'))
# df_vaccine = pd.read_csv(os.path.join(data_dir, 'vaccine_daily.csv'))
# df_infection = pd.read_csv(os.path.join(data_dir, 'infection_daily.csv'))

# # merge
# df = df.merge(df_infection, how='left', on='Date')
# df = df.merge(df_death, how='left', on='Date')
# df = df.merge(df_vaccine, how='left', on='Date')

# df = df.fillna(0)
# df['Infection'] = df['Infection'].astype(int)
# df['Death'] = df['Death'].astype(int)
# df['Vaccine'] = df['Vaccine'].astype(int)

# # plot the curves for death
# fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)
# twin = ax.twinx()
# p1, = ax.plot(df['Date'], df['POSITIVE'], "chocolate", label="POSITIVE")
# p2, = ax.plot(df['Date'], df['NEGATIVE'], "teal", label="NEGATIVE")
# p3, = twin.plot(df['Date'], df['Death'], "r", label="Death")

# ax.set_xlabel('Date', fontsize=20)
# ax.set_ylabel('Count of tweets', fontsize=20)
# twin.set_ylabel('Death', fontsize=20)
# ax.set_xticks(date_index_list)
# ax.set_xticklabels(mark_date_list, fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.legend(handles=[p1, p2, p3], fontsize=20)
# ax.set_title('Comparison between daily Tweet sentiment and death case')
# # import pdb;pdb.set_trace()
# corr_pos = np.corrcoef(df['POSITIVE'].to_list(), df['Death'].to_list())[1,0]
# corr_neg = np.corrcoef(df['NEGATIVE'].to_list(), df['Death'].to_list())[1,0]
# plt.tight_layout()
# plt.savefig(os.path.join(result_root_dir, f'senti_death_{title}.png'))
# plt.close()

# print(r'r_positive_death=%.3f' % (corr_pos))
# print(r'r_negative_death=%.3f' % (corr_neg))

# # plot the curves for vaccine
# fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)
# twin = ax.twinx()
# p1, = ax.plot(df['Date'], df['POSITIVE'], "chocolate", label="POSITIVE")
# p2, = ax.plot(df['Date'], df['NEGATIVE'], "teal", label="NEGATIVE")
# p3, = twin.plot(df['Date'], df['Vaccine'], "g", label="Vaccination")

# ax.set_xlabel('Date', fontsize=20)
# ax.set_ylabel('Count of tweets', fontsize=20)
# twin.set_ylabel('Vaccination', fontsize=20)
# ax.set_xticks(date_index_list)
# ax.set_xticklabels(mark_date_list, fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.legend(handles=[p1, p2, p3], fontsize=20)
# ax.set_title('Comparison between daily Tweet sentiment and vaccination')

# corr_pos = np.corrcoef(df['POSITIVE'].to_list(), df['Vaccine'].to_list())[1,0]
# corr_neg = np.corrcoef(df['NEGATIVE'].to_list(), df['Vaccine'].to_list())[1,0]

# plt.tight_layout()
# plt.savefig(os.path.join(result_root_dir, f'senti_vaccine_{title}.png'))
# plt.close()

# print(r'r_positive_vaccine=%.3f' % (corr_pos))
# print(r'r_negative_vaccine=%.3f' % (corr_neg))

# # plot the curves for infection
# fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)
# twin = ax.twinx()
# p1, = ax.plot(df['Date'], df['POSITIVE'], "chocolate", label="POSITIVE")
# p2, = ax.plot(df['Date'], df['NEGATIVE'], "teal", label="NEGATIVE")
# p3, = twin.plot(df['Date'], df['Infection'], "y", label="Infection")

# ax.set_xlabel('Date', fontsize=20)
# ax.set_ylabel('Count of tweets', fontsize=20)
# twin.set_ylabel('Infection', fontsize=20)
# ax.set_xticks(date_index_list)
# ax.set_xticklabels(mark_date_list, fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.legend(handles=[p1, p2, p3], fontsize=20)
# ax.set_title('Comparison between daily Tweet sentiment and infeaction')
# corr_pos = np.corrcoef(df['POSITIVE'].to_list(), df['Infection'].to_list())[1,0]
# corr_neg = np.corrcoef(df['NEGATIVE'].to_list(), df['Infection'].to_list())[1,0]

# plt.tight_layout()
# plt.savefig(os.path.join(result_root_dir, f'senti_infection_{title}.png'))
# plt.close()

# print(r'r_positive_infection=%.3f' % (corr_pos))
# print(r'r_negative_infection=%.3f' % (corr_neg))