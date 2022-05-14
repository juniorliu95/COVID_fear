# divide the emotion tweets by keywords
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

title = "vaccine"

root_dir = os.getcwd() 
output_dir = os.path.join(root_dir, 'results', 'image')
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

# kansen_keywords = ["感染", "陽性", "重症", "infect","流行", "後遺症", "蔓延", "発症", "変異", "崩壊", "症例"]
# side_keywords = ["side effect", "副作用", "副反応", "有害", "アレルギー", "アナフィラキシー", "安全性", "safety", "心筋炎", "心膜炎", "不安", "血栓", "毒", ["接種", "危険"], ["接種", "死"], ["ワクチン", "危険"], "被害", "有害", "マイクロチップ", "不妊", "妊娠", "妊婦"]
kansen_keywords = ["感染", "死亡", "日本", "デルタ", "リスク", "政府", "医療", "重症"]
side_keywords = ["副反応", "変異", "モデルナ", "予約", "ファイザー", "注射", "病院", "医師"]
rumor_keywords = ["毒", ["接種", "危険"], ["接種", "死"], ["ワクチン", "危険"], "被害", "有害", "マイクロチップ", "不妊", "妊娠", "妊婦"]
keyword_dic = {"infection":kansen_keywords, "vaccine_confidence": side_keywords}

emotion_dir = os.path.join(root_dir, 'results', "result_csv_" + title)
trans_dir = os.path.join(root_dir, 'results',"translation_csv_" + title)
text_dir = os.path.join(root_dir, 'results', title)

start_date = "2021-02-01"
end_date = "2021-09-30"
date_list = list(sorted(os.listdir(text_dir))) # dates
s_ind = date_list.index(start_date)
e_ind = date_list.index(end_date)

#%%
fear_list = []

for date in date_list[s_ind: e_ind+1]:
    print(date)

    emotion_file = os.path.join(emotion_dir, "_".join(date.split("-")) + ".csv")
    trans_file = os.path.join(trans_dir, "_".join(date.split("-")) + ".csv")
    df = pd.read_csv(emotion_file)
    df.drop(df.columns[0], axis=1, inplace=True)
    df_t = pd.read_csv(trans_file)
    df_t.drop(df_t.columns[0], axis=1, inplace=True)
    df = df.merge(df_t, on="Text")
    df['ID'] = df['ID'].apply(str)
    text_list = []
    for id in df["ID"].to_list():
        text = None
        text_file = os.path.join(text_dir, date, str(id) + ".txt")
        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                text = f.readline()
                f.close()
        text_list.append([str(id), text])
    df_text = pd.DataFrame(text_list, columns=["ID", "Text_jp"])
    df = df.merge(df_text, on="ID")
    df = df.dropna(subset=['Text_jp'])
    df = df.drop_duplicates(subset=['ID'])
    
    # get the fear tweets
    infection_count = 0
    vaccine_confidence_count = 0
    other_count = 0
    fear_text_list = []
    other_text_list = []

    for i in range(len(df)):
        fear = df.iloc[i, 2]
        if fear == 0:
            continue

        if fear != max(df.iloc[i, :6]):
            continue

        text = df["Text_jp"].iloc[i]

        fear_text_list.append([str(df["ID"].iloc[i]), text])
        mark_infection, mark_vaccine_confidence = False, False
        for keyword in keyword_dic["infection"]:
            if isinstance(keyword, list):
                mark_temp = True
                for key in keyword:
                    if key not in text:
                        mark_temp = False
                        break
                if mark_temp:
                    mark_vaccine_confidence = True
            elif keyword in text:
                mark_infection = True
        for keyword in keyword_dic["vaccine_confidence"]:
            if isinstance(keyword, list):
                mark_temp = True
                for key in keyword:
                    if key not in text:
                        mark_temp = False
                        break
                if mark_temp:
                    mark_vaccine_confidence = True
            elif keyword in text:
                mark_vaccine_confidence = True
        if not mark_vaccine_confidence and not mark_infection:
            other_count += 1
            other_text_list.append([str(df["ID"].iloc[i]), text])
        else:
            if mark_vaccine_confidence:
                vaccine_confidence_count += 1
            if mark_infection:
                infection_count += 1
    
    df_text_fear = pd.DataFrame(fear_text_list, columns=["ID", "Text"])
    df_text_fear.to_csv(os.path.join(root_dir, "results", "emotions", "fear", date+".csv"))

    df_text_fear_other = pd.DataFrame(other_text_list, columns=["ID", "Text"])
    df_text_fear_other.to_csv(os.path.join(root_dir, "results", "emotions", "fear_other", date+".csv"))
    
    del df_text_fear, df_text_fear_other

    fear_list.append([date, infection_count, vaccine_confidence_count, other_count])

    del df, df_t, df_text

df_fear = pd.DataFrame(fear_list, columns=["Date", "Infection", "Vaccine_confidence", "Other"])

#%%
# draw image
import seaborn as sb

# get the index of each 1st and 6.30
mark_date_list = ["2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-09-30"]
date_df_list = df_fear.Date.to_list()
date_index_list = [date_df_list.index(date) for date in mark_date_list]

fig, ax = plt.subplots(figsize=(20, 10))
sb.lineplot(data=df_fear, ax=ax)

ax.set_ylabel("Count of tweets", fontsize=20)
ax.set_xlabel("Date", fontsize=20)
ax.set_xlim(0, len(df_fear))
ax.set_xticks(date_index_list)
ax.set_xticklabels(mark_date_list, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)
plt.savefig(os.path.join(root_dir, "results", "image", "emotion_fear_keywords.png"))
plt.close()

# %%
plt.figure(figsize=(20, 10))
plt.plot(df_fear["Date"], df_fear["Infection"]/df_fear["Vaccine_confidence"])
plt.axhline(1, ls="--")
plt.xticks(date_index_list, mark_date_list, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Date", fontsize=20)
plt.ylabel("infection/side effect", fontsize=20)
plt.xlim(0, len(df_fear["Date"]))
plt.savefig(os.path.join(root_dir, "results", "image", "emotion_fear_ratio.png"))
plt.close()

# %%
