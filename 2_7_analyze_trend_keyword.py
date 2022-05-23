"""
The correlations between unigrams and vaccination.
"""

#%%
import os
import datetime
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags

root_dir = os.getcwd() 
output_dir = os.path.join(root_dir, 'results', 'image')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_date = "2021-02-01"
end_date = "2021-09-30"
dates = pd.date_range(start_date, end_date, freq='D')
dates = dates.strftime('%Y-%m-%d').to_list()

## ['感染', '日本', '予約', 'ファイザー', '会場', '変異']
trans_dict = {"感染":"infection", "予約":"reserve", "":"Japan", "ファイザー":"Pfizer","会場":"venue", "変異":"mutation"}
# keyword_list = ["感染", "予約", "会場"]
keyword_list = ["予約", "会場"]
# keyword_list = ["感染", "予約"]
# keyword_list = ["感染", "会場"]
# keyword_list = ["予約"]
# keyword_list = ["会場"]
# keyword_list = ["感染"]

keyword = "+".join(keyword_list)
keyword_trans = "+".join([trans_dict[key] for key in keyword_list])

keyword_dir = os.path.join(root_dir, 'results', "keywords", keyword)
# csv_file_list = list(sorted(os.listdir(keyword_dir))) # the csv files
csv_file = os.path.join(keyword_dir, keyword + ".csv")

#%%
# get the tweets number of keywords
df = None

csv_file_path = os.path.join(keyword_dir, csv_file)
if df is None:
    df = pd.read_csv(csv_file_path)
else:
    df_temp = pd.read_csv(csv_file_path)
    df = df.append(df_temp, ignore_index=True)
df = df.iloc[:,1:]
df = df.groupby(by=["Date"],as_index=False).count()
df.Date = df.Date.apply(lambda x: x.replace("/", "-"))

#%%
# get the full date
df_date = pd.DataFrame(dates, columns=["Date"])
df = df_date.merge(df, how="left", on="Date")

df.fillna(0, inplace=True)
df['ID'] = df['ID'].apply(int)

#%%
# add the data for the vaccination
df_vaccine = pd.read_csv(os.path.join(root_dir, 'data', 'vaccine_daily.csv'), index_col=False)
df_vaccine["Vaccine"] = df_vaccine["Vaccine"].apply(int)
df = df.merge(df_vaccine,how="left",on='Date')
df = df.fillna(0)
df["Vaccine"] = df["Vaccine"].apply(int)
del df_vaccine

#%%
# plot the tweet trending of vaccine, and keywords
# Make the Fig 3 in the paper
fig, ax = plt.subplots(figsize=(20, 10))
fig.subplots_adjust(right=0.75)
twin = ax.twinx()

ax.set_title(f'Tendency of tweets keywords', fontsize=20)
p1, = ax.plot(df.Date.to_list(), df.ID.values, c=(254/255.,129/255.,125/255.), linewidth=2, label=keyword_trans)
p2, = twin.plot(df.Date.to_list(), df['Vaccine'], c=(129/255.,184/255.,223/255.), linewidth=2, label="vaccination")

# get the index of each 1st and 9.30
mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_df_list = df.Date.to_list()
date_index_list = [date_df_list.index(date) for date in mark_date_list]

ax.set_xticks(date_index_list)
ax.set_xticklabels(mark_date_list)
ax.tick_params(axis='x', labelsize=20)
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Count of tweets',fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, max(df.ID.values)+5)
ax.set_xlim(0, date_index_list[-1])
twin.set_ylim(0, max(df.Vaccine.values)+5)
twin.set_ylabel('Daily vaccination', fontsize=20)
twin.tick_params(axis='y', labelsize=15)

ax.legend(handles=[p1, p2], fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'trend_of_tweet_keywords_{keyword}.png'))
plt.close()

print(f"correlation vaccination and {keyword_trans}: ", pearsonr(df.ID, df.Vaccine))

print('Total trend saved...')
# %%
# print the cross correlation results
correlation = np.correlate(df.ID, df.Vaccine, mode='same')
lags = correlation_lags(len(df.ID), len(df.Vaccine), mode="same")

max_corr = np.argmax(correlation)
lag = lags[max_corr]
print(keyword, lag, max_corr)

# %%
