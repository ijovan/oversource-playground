import json
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from questions import Questions


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


def print_top_words(model, feature_names, n_top_words, topics):
    for topic_idx, topic in enumerate(model.components_):
        match = []
        for key, value in topics.items():
            if value == topic_idx:
                match.append(key)
        print("---> Topic #" + str(topic_idx) + ": " + ", ".join(match))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def classify(texts, tags):
    fit(MultinomialNB(), texts, tags)
    fit(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                      random_state=42, max_iter=5, tol=None), texts, tags)


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


N_COMPONENTS = 16


def lda(questions, languages):
    topics = {}

    for language in languages:
        topics[language] = {}
        for i in range(0, N_COMPONENTS):
            topics[language][i] = 0

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=1000,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(questions.texts())
    lda = LatentDirichletAllocation(n_components=N_COMPONENTS, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0, verbose=1)
    lda.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    tags = questions.tags()

    i = 0
    for row in lda.transform(tfidf):
        topics[tags[i]][numpy.argmax(row)] += 1
        i += 1

    counts = {}
    for i in range(0, N_COMPONENTS):
        counts[i] = 0
        for topic in topics.values():
            counts[i] += topic[i]

    for value in topics.values():
        for i in value:
            if value[i] > 0:
                value[i] = value[i] / counts[i]

    print(topics)
    final_topics = {}
    for key, value in topics.items():
        final_topics[key] = int(numpy.argmax(list(value.values())))

    print_top_words(lda, tfidf_feature_names, 20, final_topics)


languages = json.loads(open("../languages.json", "r").read())
questions = Questions()

# classify(questions.texts(), questions.tags())

# languages_sentiment(questions, languages)

lda(questions, languages)
