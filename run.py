import csv
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def read(filename):
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        return [row for row in reader]


def fit(classifier, texts, tags):
    train_texts = texts[:len(texts)//2]
    test_texts = texts[len(texts)//2:]
    train_tags = tags[:len(tags)//2]
    test_tags = tags[len(tags)//2:]

    clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', classifier)])

    clf.fit(train_texts, train_tags)
    predicted = clf.predict(test_texts)
    print(metrics.classification_report(test_tags, predicted))


def classify(texts, tags):
    fit(MultinomialNB(), texts, tags)
    fit(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                      random_state=42, max_iter=5, tol=None), texts, tags)


def sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()

    i = 0
    compound = 0
    neu = 0
    neg = 0
    pos = 0

    for text in texts:
        score = analyzer.polarity_scores(text)
        i += 1
        compound += score['compound']
        neu += score['neu']
        neg += score['neg']
        pos += score['pos']

    return {'compound': compound / i, 'neutral': neu / i,
            'negative': neg / i, 'positive': pos / i}


def language_filter(questions, language):
    return list(filter(lambda question:
                       question[2] == language, questions))


def texts(questions):
    return list(map(lambda question: question[3], questions))


def tags(questions):
    return list(map(lambda question: question[2], questions))


def languages_sentiment(questions, languages):
    for language in languages:
        lang_questions = language_filter(questions, language)
        texts = list(map(lambda question: question[3], lang_questions))
        print("{0}: {1}".format(language, sentiment(texts)))


# languages = json.loads(open("../languages.json", "r").read())
questions = read("../language_questions.csv")

texts = texts(questions)
tags = tags(questions)

classify(texts, tags)

# languages_sentiment(questions, languages)
