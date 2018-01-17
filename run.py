import json
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from questions import Questions


def fit(classifier, texts, tags):
    train_texts, test_texts, train_tags, test_tags = \
            train_test_split(texts, tags, test_size=0.2)

    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),
                                             stop_words='english')),
                   ('tfidf', TfidfTransformer()),
                   ('clf', classifier)])

    clf.fit(train_texts, train_tags)
    predicted = clf.predict(test_texts)
    print(metrics.classification_report(test_tags, predicted))


def print_top_words(model, feature_names, n_top_words, topics):
    for topic_idx, topic in enumerate(model.components_):
        match = []
        for key, value in topics.items():
            if value == topic_idx:
                match.append(key)
        if match != []:
            print("---> Topic #" + str(topic_idx) + ": " + ", ".join(match))
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def classify(texts, tags):
    fit(MultinomialNB(), texts, tags)
    fit(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                      random_state=42, max_iter=5, tol=None), texts, tags)


def find_topics(questions, languages):
    decompose(questions, languages,
              LatentDirichletAllocation(n_components=N_COMPONENTS,
                                        max_iter=10,
                                        learning_method='online',
                                        learning_offset=50,
                                        random_state=0, verbose=1))
    # decompose(questions, languages,
    #           NMF(n_components=N_COMPONENTS, random_state=1,
    #               beta_loss='kullback-leibler', solver='mu',
    #               max_iter=1000, alpha=.1, l1_ratio=.5, verbose=1))


def sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()

    count = len(texts)
    compound, neutral, negative, positive = 0, 0, 0, 0

    for text in texts:
        score = analyzer.polarity_scores(text)
        compound += score['compound']
        neutral += score['neu']
        negative += score['neg']
        positive += score['pos']

    return {'compound': compound / count, 'neutral': neutral / count,
            'negative': negative / count, 'positive': positive / count}


def languages_sentiment(questions, languages):
    for language in languages:
        texts = questions.for_language(language).texts()
        print("{0}: {1}".format(language, sentiment(texts)))


N_COMPONENTS = 20


def decompose(questions, languages, algorithm):
    topics = {}

    for language in languages:
        topics[language] = {}
        for i in range(0, N_COMPONENTS):
            topics[language][i] = 0

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=2000,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(questions.texts())
    algorithm.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    tags = questions.tags()

    i = 0
    for row in algorithm.transform(tfidf):
        topics[tags[i]][np.argmax(row)] += 1
        i += 1

    counts = {}
    for i in range(0, N_COMPONENTS):
        counts[i] = 0
        for topic in topics.values():
            counts[i] += topic[i]

    for value in topics.values():
        for i in value:
            if value[i] > 0:
                if counts[i] / len(topics.items()) > 0.5 / N_COMPONENTS:
                    value[i] = value[i] / counts[i]
                else:
                    value[i] = 0

    final_topics = {}
    for key, value in topics.items():
        final_topics[key] = int(np.argmax(list(value.values())))

    print_top_words(algorithm, tfidf_feature_names, 15, final_topics)


LIMIT = 800


def classify_even(questions, languages):
    texts = []
    tags = []
    for lang in languages:
        texts += questions.for_language(lang).texts()[:LIMIT]
        tags += questions.for_language(lang).tags()[:LIMIT]
    classify(texts, tags)


def dateparse(stamp):
    return pd.datetime.fromtimestamp(float(stamp)).date()


def time_series():
    rcParams['figure.figsize'] = 15, 6

    questions = pd.read_csv('../ts.csv', delimiter=',', parse_dates=True,
                            date_parser=dateparse, index_col='created_at',
                            names=['created_at', 'comment_count'],
                            header=None)

    questions = questions['2013':'2017'].sort_index()
    questions = \
        questions.groupby(lambda x: x)['comment_count'].agg(['sum'])

    ts = questions['sum']

    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    plt.plot(ts, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.show()


languages = json.loads(open("../languages.json", "r").read())
questions = Questions()

######################################################################

classify(questions.texts(), questions.tags())

# languages_sentiment(questions, languages)

# find_topics(questions, languages)

# time_series()
