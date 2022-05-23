# COVID_fear

This project contains the code for our paper: "Fear of Infection and Sufficient Vaccine Reservation Information Might Drive Rapid Coronavirus Disease 2019 Vaccination in Japan: Evidence from Twitter Analysis".

---

## Data aquisition
The Twitter dataset is available at https://github.com/thepanacealab/covid19_twitter. The notebook `COVID_19_dataset_Tutorial.ipynb` in the repository was adjusted to only collect Japanese tweets and applied day by day to collect the daily tweets.

The daily vaccination data was collected from the official website of the Prime Minister’s Office of Japan (PMOJ).

## Requirements
* Twitter API key
### packages:
* spacy
* ginza
* numpy
* scikit-learn
* pandas
* matplotlib
* japanize_matplotlib
* wordcloud
* googletrans==4.0.0-rc1

## Reproducing images in the paper
### Fig 1
```
python 2_3_top_words_condition.py
```
### Fig 2
```
python 2_4_top_biwords_condition.py
```
### Fig 3
```
python 2_7_analyze_trend_keyword.py
```
### Fig 4
```
python 3_1_sentiment.py
```
### Fig 5
```
python 3_3_draw_emotion.py
```
### Fig 6-7
```
python 4_2_LDA_fear.py
```
### Appendix Fig 1
```
python 4_1_LDA_fear_topic_num.py
```

## Reference
Banda, Juan M., et al. "A large-scale COVID-19 Twitter chatter dataset for open scientific research—an international collaboration." Epidemiologia 2.3 (2021): 315-324.

## Citing this repository
