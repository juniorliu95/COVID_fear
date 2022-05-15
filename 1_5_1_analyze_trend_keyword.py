"""
The correlations between unigrams and vaccination.
"""
# ! after 2_5_1 or 3_5

#%%
import os

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
csv_file_list = [os.path.join(keyword_dir, keyword + ".csv")]

#%%
# get the tweets number of keywords
df = None
for csv_file in csv_file_list:
    csv_file_path = os.path.join(keyword_dir, csv_file)
    if df is None:
        df = pd.read_csv(csv_file_path)
    else:
        df_temp = pd.read_csv(csv_file_path)
        df = df.append(df_temp, ignore_index=True)
df = df.iloc[:,1:]
df = df.groupby(by=["Date"],as_index=False).count()
df = df.iloc[:,[0,2]]

# get the full date
dates = sorted(os.listdir(os.path.join(root_dir, 'results','vaccine')))
start = dates.index(start_date)
end = dates.index(end_date)
dates = dates[start:end+1]
dates = [[date, 0] for date in dates]

df_date = pd.DataFrame(dates, columns=["Date", "Text"])
df = df_date.merge(df,how="left", on="Date")

df.fillna(0, inplace=True)
df['Text'] = df['Text_x'] + df['Text_y']
df = df.iloc[:,[0,3]]
df['Text'] = df['Text'].apply(int)

#%%
# add the data for the vaccination
df_vaccine = pd.read_csv(os.path.join(root_dir, 'data', 'vaccine_daily.csv'))

df = df.merge(df_vaccine,how="left",on='Date')
df = df.fillna(0)
df["Vaccine"] = df["Vaccine"].apply(int)
del df_vaccine

#%%
# add the norm vaccination
df_norm_vaccine = pd.read_csv(os.path.join(root_dir, 'data', 'norm_vaccine_daily.csv'))
df_norm_vaccine = df_norm_vaccine.rename(columns={'Vaccine':'Vaccine_norm'})
df = df.merge(df_norm_vaccine,how="left",on='Date')
df = df.fillna(0)
df["Vaccine_norm"] = df["Vaccine_norm"].apply(int)
del df_norm_vaccine


#%%
# plot the tweet trending of vaccine, olympics and vaccine + olympics
fig, ax = plt.subplots(figsize=(20, 10))
fig.subplots_adjust(right=0.75)
twin = ax.twinx()

ax.set_title(f'Tendency of tweets keywords', fontsize=20)
p1, = ax.plot(df.Date.to_list(), df.Text.values, c=(254/255.,129/255.,125/255.), linewidth=2, label=keyword_trans)
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
ax.set_ylim(0, max(df.Text.values)+5)
ax.set_xlim(0, date_index_list[-1])
twin.set_ylim(0, max(df.Vaccine.values)+5)
twin.set_ylabel('Daily vaccination', fontsize=20)
twin.tick_params(axis='y', labelsize=15)

ax.legend(handles=[p1, p2], fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'trend_of_tweet_keywords_{keyword}.png'))
plt.close()

print(f"correlation vaccination and {keyword_trans}: ", pearsonr(df.Text, df.Vaccine))

print('Total trend saved...')

# %%
# print the cross correlation results
correlation = np.correlate(df.Text, df.Vaccine, mode='same')
lags = correlation_lags(len(df.Text), len(df.Vaccine), mode="same")

max_corr = np.argmax(correlation)
lag = lags[max_corr]
print(keyword, lag, max_corr)
