"""Get the trend of number of tweets associated with vaccine"""
#%%
from datetime import date
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import csv

title = 'vaccine'
# title = 'vaccine_and_olympics'

start_date = '2021-02-01'
end_date = '2021-10-01'

root_dir = os.getcwd()
keyword_dir = os.path.join(root_dir, 'results', title)
folder_list = list(sorted(os.listdir(keyword_dir))) # this is also the date
data_dir = os.path.join(root_dir, 'data')
output_dir = os.path.join(root_dir, 'results', 'image')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# the numeber of tweets
freq_list = []
for f_ind, folder_name in enumerate(folder_list):
    files_path = os.path.join(keyword_dir, folder_name)
    file_names = os.listdir(files_path)
    freq_list.append(len(file_names))


# save the file for the number of tweets
df_tweet = pd.DataFrame(np.array([folder_list, freq_list]).T, columns=["Date", "Tweets"])
df_tweet.to_csv(os.path.join(data_dir, f'daily_tweets_{title}.csv'), index=False)

#%%
# daily infection
infect_daily = []
infect_date = []
with open(os.path.join(data_dir, 'newly_confirmed_cases_daily.csv'), 'r') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='|')
    for row in spamreader:
        if '/' not in row[0]:
            continue

        if row[1]!='ALL':
            continue
        
        year, month, day = row[0].split('/')
        row[0] = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))
        infect_date.append(row[0])
        infect_daily.append(int(row[2]))
df_infect = pd.DataFrame(list(zip(infect_date, infect_daily)), columns=['Date', 'Infection'])
if start_date is not None:
    los = df_infect.index[df_infect['Date']==start_date].tolist()[0]
    df_infect = df_infect.iloc[los:, :]
if end_date is not None:
    loe = df_infect.index[df_infect['Date']==end_date].tolist()[0]
    df_infect = df_infect.iloc[:loe, :]
df_infect.to_csv(os.path.join(data_dir, 'infection_daily.csv'), index=False)
print('infection saved...')

#%%
# daily death
death_daily = []
death_date = []
last_death_total = 0
with open(os.path.join(data_dir, 'deaths_cumulative_daily.csv'),'r') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='|')
    for row in spamreader:
        if '/' not in row[0]:
            continue

        if row[1]!='ALL':
            continue

        year, month, day = row[0].split('/')
        row[0] = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))
        death_date.append(row[0])
        death_daily.append(int(row[2])-last_death_total)
        last_death_total = int(row[2])

df_death = pd.DataFrame(np.array([death_date, death_daily]).T, columns=['Date','Death'])
if start_date is not None:
    los = df_death.index[df_death['Date']==start_date].tolist()[0]
    df_death = df_death.iloc[los:, :]
if end_date is not None:
    loe = df_death.index[df_death['Date']==end_date].tolist()[0]
    df_death = df_death.iloc[:loe, :]

df_death.to_csv(os.path.join(data_dir, 'death_daily.csv'), index=False)
print('death saved...')
# only use the data from the beginning of analysis
death_daily = death_daily[death_date.index(folder_list[0]):]

#%%
# daily vaccination
df_vaccine = pd.read_csv(os.path.join(data_dir, 'raw_vaccination.csv'))
for i in range(len(df_vaccine)):
    temp = df_vaccine.loc[i, "Date"]
    temp = list(map(int, temp.split("/")))
    temp = '{}-{:02d}-{:02d}'.format(temp[2], temp[0], temp[1])
    df_vaccine.loc[i, "Date"] = temp 
df_vaccine = df_vaccine.sort_values(by=['Date'])

if start_date is not None and start_date in df_vaccine.Date.to_list():
    los = df_vaccine.index[df_vaccine['Date']==start_date].tolist()[0]
    df_vaccine = df_vaccine.iloc[los:, :]
if end_date is not None and end_date in df_vaccine.Date.to_list():
    loe = df_vaccine.index[df_vaccine['Date']==end_date].tolist()[0]
    df_vaccine = df_vaccine.iloc[:loe, :]
df_vaccine.to_csv(os.path.join(data_dir, 'vaccine_daily.csv'), index=False)
print('vaccination saved...')



#%%
# all data
df = df_tweet.merge(df_infect, how='left', on='Date')
df = df.merge(df_death, how='left', on='Date')
df = df.merge(df_vaccine, how='left', on='Date')

df = df.fillna(0)
df['Tweets'] = df['Tweets'].astype(int)
df['Infection'] = df['Infection'].astype(int)
df['Death'] = df['Death'].astype(int)
df['Vaccine'] = df['Vaccine'].astype(int)

#%%
# draw images

# index of the point when vaccination start
olym_start = int(df.index[df['Date']=='2021-07-23'].to_list()[0])
olym_end = int(df.index[df['Date']=='2021-09-05'].to_list()[0])

# plot the curves for death
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin = ax.twinx()
p1, = ax.plot(df['Date'], df['Tweets'], "b-", label="Tweets")
p2, = twin.plot(df['Date'], df['Death'], "r-", label="Death")
ax.axvline(olym_start, c='k', ls='--')
ax.axvline(olym_end, c='k', ls='--')
ax.add_patch(matplotlib.patches.Rectangle((olym_start,0), olym_end-olym_start, 1600, color="gray", alpha=.1))

ax.set_xlabel('Date')
ax.set_ylabel('Count of tweets')
twin.set_ylabel('Death')
ax.set_xticks(np.arange(len(df['Date']))[::5])
ax.set_xticklabels(df['Date'][::5], rotation=45)
ax.legend(handles=[p1, p2], fontsize=10)
ax.set_title(f'Comparison between daily number of {title}-related Tweet\n and death case')
corr = np.corrcoef(df['Tweets'], df['Death'])[1,0]
corr_olym = np.corrcoef(df['Tweets'].to_list()[olym_start:olym_end], df['Death'].to_list()[olym_start:olym_end])[1,0]
print("correlation_death_whole:{}".format(np.around(corr, 3)))
print("correlation_death_olympics:{}".format(np.around(corr_olym, 3)))

height = int(max(df["Tweets"]) * 0.9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{title}-related_tweet_death.png'))
plt.close()

# plot the curves for vaccine
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin = ax.twinx()
p1, = ax.plot(df['Date'], df['Tweets'], "b-", label="Tweets")
p2, = twin.plot(df['Date'], df['Vaccine'], "g-", label="Vaccination")
ax.axvline(olym_start, c='k', ls='--')
ax.axvline(olym_end, c='k', ls='--')
ax.add_patch(matplotlib.patches.Rectangle((olym_start,0), olym_end-olym_start, 1600, color="gray", alpha=.1))

ax.set_xlabel('Date')
ax.set_ylabel('Count of tweets')
twin.set_ylabel('Daily vaccination')
ax.set_xticks(np.arange(len(df['Date']))[::5])
ax.set_xticklabels(df['Date'][::5], rotation=45)
ax.legend(handles=[p1, p2], fontsize=10)
ax.set_title(f'Comparison between daily number of {title}-related Tweet\n and vaccination')
corr = np.corrcoef(df['Tweets'], df['Vaccine'])[1,0]
corr_olym = np.corrcoef(df['Tweets'].to_list()[olym_start:olym_end], df['Vaccine'].to_list()[olym_start:olym_end])[1,0]
print("correlation_vaccine_whole:{}".format(np.around(corr, 3)))
print("correlation_vaccine_olympics:{}".format(np.around(corr_olym, 3)))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{title}-related_tweet_vaccine.png'))
plt.close()

# plot the curves for infection
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin = ax.twinx()
p1, = ax.plot(df['Date'], df['Tweets'], "b-", label="Tweets")
p2, = twin.plot(df['Date'], df['Infection'], "y-", label="Infection")
ax.axvline(olym_start, c='k', ls='--')
ax.axvline(olym_end, c='k', ls='--')
ax.add_patch(matplotlib.patches.Rectangle((olym_start,0), olym_end-olym_start, 1600, color="gray", alpha=.1))

ax.set_xlabel('Date')
ax.set_ylabel('Count of tweets')
twin.set_ylabel('Daily infection')
ax.set_xticks(np.arange(len(df['Date']))[::5])
ax.set_xticklabels(df['Date'][::5], rotation=45)
ax.legend(handles=[p1, p2], fontsize=10)
ax.set_title(f'Comparison between daily number of {title}-related Tweet\n and infection')
corr = np.corrcoef(df['Tweets'], df['Infection'])[1,0]
corr_olym = np.corrcoef(df['Tweets'].to_list()[olym_start:olym_end], df['Infection'].to_list()[olym_start:olym_end])[1,0]
print("correlation_infection_whole:{}".format(np.around(corr, 3)))
print("correlation_infection_olympics:{}".format(np.around(corr_olym, 3)))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{title}-related_tweet_infection.png'))
plt.close()


# plot the tweet trending
plt.figure(figsize=(20, 10))
plt.title(f'Tendency of {title}-related tweets',fontsize=20)

plt.plot(np.arange(len(freq_list)), freq_list, 'b-')
plt.xticks(np.arange(len(freq_list), step=5), folder_list[::5])
plt.xlabel('Date',fontsize=20)
plt.ylabel('Count of tweets',fontsize=20)
plt.ylim(0, 1600)
ax = plt.gca()
ax.axvline(olym_start, c='k', ls='--')
ax.axvline(olym_end, c='k', ls='--')
ax.add_patch(matplotlib.patches.Rectangle((olym_start,0), olym_end-olym_start, 1600, color="gray", alpha=.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'num_of_{title}-related_tweet.png'))
plt.close()
print('Total saved...')


# plot the tweet trending of vaccine, olympics and vaccine + olympics

plt.figure(figsize=(20, 10))
plt.title(f'Tendency of tweets keywords',fontsize=20)

keywords = ["vaccine", "olympics", "vaccine_and_olympics"]
for keyword in keywords:
    keyword_dir = os.path.join(root_dir, 'results', keyword)
    folder_list = list(sorted(os.listdir(keyword_dir))) # this is also the date
    # the numeber of tweets
    freq_list = []
    for f_ind, folder_name in enumerate(folder_list):
        files_path = os.path.join(keyword_dir, folder_name)
        file_names = os.listdir(files_path)
        freq_list.append(len(file_names))
    plt.plot(np.arange(len(freq_list)), freq_list, label=keyword)

plt.xticks(np.arange(len(freq_list), step=5), folder_list[::5])
plt.xlabel('Date',fontsize=20)
plt.ylabel('Count of tweets',fontsize=20)
plt.ylim(0, 1600)
ax = plt.gca()
ax.axvline(olym_start, c='k', ls='--')
ax.axvline(olym_end, c='k', ls='--')
ax.add_patch(matplotlib.patches.Rectangle((olym_start,0), olym_end-olym_start, 1600, color="gray", alpha=.1))
plt.tight_layout()
plt.legend(fontsize=20)
plt.savefig(os.path.join(output_dir, f'trend_of_tweet_keywords.png'))
plt.close()
print('Total trend saved...')