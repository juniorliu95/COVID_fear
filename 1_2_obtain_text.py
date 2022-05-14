"""1. Obtain the text for sentiment analysis from all the documents"""

import json
import os
import codecs

root_dir = os.getcwd()
data_dir = os.path.join(root_dir, '../covid19_twitter', 'dailies')
start_date = [2021, 9, 30] # the first day to record
end_date = [2021, 10, 1] # The day after the last day
start_date_folder = '{}-{:02d}-{:02d}'.format(*start_date)
end_date_folder = '{}-{:02d}-{:02d}'.format(*end_date)
all_date_list = list(sorted(os.listdir(data_dir)))

# output root path
result_root_dir = os.path.join(root_dir, 'results', 'text')
if not os.path.exists(result_root_dir):
	os.makedirs(result_root_dir)


key = False # the key controlling the reading of the folder
for d_ind, date in enumerate(all_date_list):
	# judge if the date is within the selected date range
	if '-' not in date:
		continue
	if date == start_date_folder:
		key = True
	if date == end_date_folder:
		key = False
	if key == False:
		continue
	print(date)

	f_day = int(date.split('-')[-1]) # the day of the folder. May contain tweets of different days due to time zone

	if not os.path.exists(os.path.join(result_root_dir, date)):
		os.makedirs(os.path.join(result_root_dir, date))
	if not os.path.exists(os.path.join(result_root_dir, all_date_list[d_ind+1])):
		os.makedirs(os.path.join(result_root_dir, all_date_list[d_ind+1]))

	with open(os.path.join(data_dir, date, 'hydrated_tweets_short.json'), 'r') as f:
		for t_ind, tweet in enumerate(f):
			data = json.loads(tweet)

			# adjust the time zone
			t_day = int(data['created_at'].split()[2]) # get the day of the tweet
			text_data = codecs.encode(data['text'].replace('\n', ' '), 'utf-8', errors='replace')
			text_data = codecs.decode(text_data, 'utf-8', errors='replace')
			text_data.replace('\n',' ')

			if t_day == f_day: # if the same day
				with codecs.open(os.path.join(result_root_dir, date, data['id_str']+'.txt'), 'w', 'utf-8') as g:
					g.write(text_data)
					g.close()
			else:
				with codecs.open(os.path.join(result_root_dir, all_date_list[d_ind+1], data['id_str']+'.txt'), 'w', 'utf-8') as g:
					g.write(text_data)
					g.close()
		
