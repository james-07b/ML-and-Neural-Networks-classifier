import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from simpleclassifier import logistic_regression_thing

# Initialising Vader Sentiment
analyser = SentimentIntensityAnalyzer()


def sentiment_machine(sentence):
    score = analyser.polarity_scores(sentence)
    return score


df = pd.read_csv('data/steph.csv')
df.columns = df.columns.to_series().apply(lambda x: x.strip())

my_df = df[['tweet', 'score']]
my_df['vader_score'] = -2
my_df['vader_score'] = my_df['tweet'].apply(lambda x: 'NaN' if pd.isnull(x) else sentiment_machine(x))
my_df['compound'] = my_df['vader_score'].apply(lambda score_dict: score_dict['compound'])
my_df['comp_score'] = my_df['compound'].apply(lambda c: 1 if c > 0.05 else (0 if c < -0.05 else 2))
my_df = my_df[my_df['score'] != 2]
my_df = my_df[my_df['comp_score'] != 2]

print(my_df)
print(my_df["score"].value_counts())
print(my_df["comp_score"].value_counts())


x = my_df["score"].value_counts()

print(my_df)
print(my_df["score"].value_counts())
print(my_df["comp_score"].value_counts())


def keeping_vader_on_its_toes(test_df):
    vader_is_good_count = 0
    for index, row in test_df.iterrows():
        if row.score == row.comp_score:
            vader_is_good_count +=1

    exact_acc = (vader_is_good_count/(x.values[0] + x.values[1])) * 100
    vader_accuracy = "{:.2f}".format(exact_acc/100)
    print('Vader is this much sure ',vader_accuracy)


logistic_regression_thing('data/steph.csv')
keeping_vader_on_its_toes(my_df)




