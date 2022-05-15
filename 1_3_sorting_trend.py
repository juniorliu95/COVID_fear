"""
Get the trend of number of tweets associated with vaccine.
Sort the daily vaccination data.
"""
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
df = pd.DataFrame(np.array([folder_list, freq_list]).T, columns=["Date", "Tweets"])
df.to_csv(os.path.join(data_dir, f'daily_tweets_{title}.csv'), index=False)


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
