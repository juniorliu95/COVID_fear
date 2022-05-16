"""
Draw the curves of emotion distribution
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

keyword = "vaccine"

emotion_path_root = f"./results/result_csv_{keyword}" # path to the R results
emo_path = os.path.join("./data", "emotion") # path to save the results

trans_path_root = f"./results/translation_csv_{keyword}"
res_path = f"./results/image"
if not os.path.exists(res_path):
    os.makedirs(res_path)

file_names = os.listdir(emotion_path_root)
file_names.sort()
file_names = file_names
date = [file_name.split(".")[0] for file_name in file_names]

pos_data = []
neg_data = []
pos_list = ['anticipation', 'trust', 'joy', 'surprise']
neg_list = ['anger', 'disgust','fear','sadness']

pos_df, neg_df = None, None
if not os.path.exists(os.path.join(emo_path, "emotion_pos.csv")):
    for i, file_name in enumerate(file_names):
        if "copy" in file_name:
            continue

        emotion_path = os.path.join(emotion_path_root, file_name)
        temp = pd.read_csv(emotion_path) # read emotion
        trans_path = os.path.join(trans_path_root, file_name)
        trans_temp = pd.read_csv(trans_path)
        temp = pd.merge(temp, trans_temp, on="Text")

        senti_temp = []
        drop_index = []
        for ind in range(len(temp)):
            senti_path = os.path.join('./results/', 'sentiments_vaccine', temp["Date"][ind], str(temp["ID"][ind]) + ".txt.csv")
            if os.path.exists(senti_path):
                with open(senti_path, "r") as f:
                    senti = f.readline().split(",")[2]
                    senti_temp.append(senti)
                    f.close()
            else:
                drop_index.append(ind)
        temp = temp.drop(index=drop_index)
        temp["Sentiment"] = senti_temp

        res_temp = []
        for pos_emo in pos_list:
            valence = temp.loc[temp.Sentiment=="POSITIVE"][pos_emo].sum() / len(temp.loc[temp.Sentiment!="MIXED"].loc[temp.Sentiment!="NEUTRAL"])
            res_temp.append([date[i], pos_emo, valence])
        pos_data.extend(res_temp[:])

        res_temp = []
        for neg_emo in neg_list:
            valence = temp.loc[temp.Sentiment=="NEGATIVE"][neg_emo].sum() / len(temp.loc[temp.Sentiment!="MIXED"].loc[temp.Sentiment!="NEUTRAL"])
            res_temp.append([date[i], neg_emo, valence])
        neg_data.extend(res_temp[:])
    for pos in pos_data:
        if len(pos) == 4:
            print(pos)
    pos_df = pd.DataFrame(pos_data, columns=["Date", "Emotion", "Valence"])
    neg_df = pd.DataFrame(neg_data, columns=["Date", "Emotion", "Valence"])
    # save the results
    pos_df.to_csv(os.path.join(emo_path, "emotion_pos.csv"))
    neg_df.to_csv(os.path.join(emo_path, "emotion_neg.csv"))
else:
    pos_df = pd.read_csv(os.path.join(emo_path, "emotion_pos.csv"))
    neg_df = pd.read_csv(os.path.join(emo_path, "emotion_neg.csv"))
# import pdb;pdb.set_trace()

#%%
# index of the point when vaccination start
dates = np.unique(pos_df['Date'])
dates = [date.replace("_", "-") for date in dates]
# get the index of each 1st and 6.30
mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_index_list = [dates.index(date) for date in mark_date_list]


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=[10, 7])
ax = axes[0]
ax.set_title("Distribution of emotion valences for anticipation, trust, joy, and surprise")
sb.lineplot(data=pos_df, x="Date", y="Valence", hue="Emotion", style='Emotion', ax=ax)

ax.set_ylim(0, 2)
ax.set_xlim(0, len(dates))
ax.set_xticks(date_index_list)
ax.set_xticklabels(mark_date_list)
# ax.tick_params(axis='y', labelsize=20)
ax.legend(loc=1)

ax = axes[1]
ax.set_title("Distribution of emotion valences for fear, sadness, anger, and disgust")
sb.lineplot(data=neg_df, x="Date", y="Valence", hue="Emotion", style='Emotion', ax=ax)
ax.set_ylim(0, 2)
ax.set_xlim(0, len(dates))
ax.set_xticks(date_index_list)
ax.set_xticklabels(mark_date_list)
# ax.tick_params(axis='y', labelsize=20)
ax.legend(loc=1)

plt.tight_layout()

plt.savefig(os.path.join(res_path, "emotion.png"))


# %%
# calculate the daily average degree of valences

