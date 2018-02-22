import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from questions import Questions


def fit(classifier, texts, tags):
    train_texts, test_texts, train_tags, test_tags = \
            train_test_split(texts, tags, test_size=0.8)

    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),
                                             stop_words='english')),
                   ('tfidf', TfidfTransformer()),
                   ('clf', classifier)])

    clf.fit(train_texts, train_tags)
    predicted = clf.predict(test_texts)
    print(metrics.classification_report(test_tags, predicted))


def classify(texts, tags):
    # fit(MultinomialNB(), texts, tags)
    fit(SGDClassifier(loss='hinge', penalty='l2', verbose=True,
                      random_state=42, max_iter=10), texts, tags)


questions = Questions()

classify(questions.texts(), questions.tags())
