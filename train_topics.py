import numpy
from questions import Questions
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn import metrics


TRAIN_PERCENTAGE = 0.5

MAX_FEATURES = 3000
MIN_DF = 1e-4
TOKEN_PATTERN = "c[#\+]{1,2}|[!~][=]{1,2}|[:\-\+]{2}|[{}$@]|[()/]{2}|[A-Za-z]+"

N_COMPONENTS = 20
MAX_ITER = 5


def map_tags_components(text_components, tags):
    tag_components, tag_counts = {}, {}

    for tag in tags:
        tag_components[tag] = [0] * len(text_components[0])
        tag_counts[tag] = 0

    for index, components in enumerate(text_components):
        tag = train_tags[index]
        tag_components[tag] = numpy.add(tag_components[tag], components)
        tag_counts[tag] += 1

    for tag in tags:
        tag_components[tag] = \
            numpy.divide(tag_components[tag], tag_counts[tag])

    return tag_components


def predict(text_components, tag_components):
    predicted_tags = []

    for row in text_components:
        distances = {}

        for tag in tags:
            distances[tag] = distance.euclidean(row, tag_components[tag])

        predicted_tag = min(distances, key=distances.get)
        predicted_tags.append(predicted_tag)

    return predicted_tags


questions = Questions()
tags = numpy.unique(questions.tags())

train_texts, test_texts, train_tags, test_tags = train_test_split(
    questions.texts(), questions.tags(), test_size=(1 - TRAIN_PERCENTAGE)
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1), stop_words='english', token_pattern=TOKEN_PATTERN,
    analyzer="word", min_df=MIN_DF, max_features=MAX_FEATURES
)

train_text_vectors = vectorizer.fit_transform(train_texts)

lda = LatentDirichletAllocation(
    n_components=N_COMPONENTS, max_iter=MAX_ITER, verbose=1,
    learning_method='online', learning_offset=50, random_state=0
)

train_text_components = lda.fit_transform(train_text_vectors)

tag_components = map_tags_components(train_text_components, tags)

test_text_vectors = vectorizer.transform(test_texts)
test_text_components = lda.transform(test_text_vectors)

predicted_tags = predict(test_text_components, tag_components)

print(metrics.classification_report(test_tags, predicted_tags))
