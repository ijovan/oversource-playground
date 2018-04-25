import json
import csv
import codecs
from stack_overflow import StackOverflow
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def stack_overflow():
    return StackOverflow().pairs()

def hacker_news():
    reader = csv.reader(codecs.open("classified.csv", "rU", "utf-8"))
    return [row for row in reader]


results = [[
    "Language", "Compound", "Neutral", "Negative", "Positive", "Text"
]]

analyzer = SentimentIntensityAnalyzer()

for pair in hacker_news():
    score = analyzer.polarity_scores(pair[1])

    results.append([
        pair[0], score['compound'], score['neu'],
        score['neg'], score['pos'], pair[1]
    ])

with open('sentiments.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(results)
