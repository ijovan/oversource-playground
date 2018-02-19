import json
import numpy as np
from questions import Questions
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


N_COMPONENTS = 20


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


languages = json.loads(open("../languages.json", "r").read())
questions = Questions()

find_topics(questions, languages)
