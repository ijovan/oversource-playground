import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from questions import Questions


EVEN_LIMIT = 800


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
    fit(MultinomialNB(), texts, tags)
    fit(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                      random_state=42, max_iter=5, tol=None), texts, tags)


def classify_even(questions, languages):
    texts = []
    tags = []
    for lang in languages:
        texts += questions.for_language(lang).texts()[:EVEN_LIMIT]
        tags += questions.for_language(lang).tags()[:EVEN_LIMIT]
    classify(texts, tags)


languages = json.loads(open("../languages.json", "r").read())
questions = Questions()

classify(questions.texts(), questions.tags())
