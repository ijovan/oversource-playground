import csv
from regular_classifier import RegularClassifier
from hacker_news import HackerNews


classifier = RegularClassifier()
classifier.load()

texts = HackerNews().texts()
tags = classifier.predict(texts)

tuples = list(zip(tags, texts))

with open('classified.csv', 'w+') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for row in tuples:
        writer.writerow(row)
