import numpy
from questions import Questions
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn import metrics


TRAIN_PERCENTAGE = 0.8

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 3000
MIN_DF = 1e-5
TOKEN_PATTERN = "c[#\+]{1,2}|[!~][=]{1,2}|[:\-\+]{2}|[{}$@]|[()/]{2}|[A-Za-z]+"

N_COMPONENTS = 15
MAX_ITER = 5


def map_tag_distributions(text_components, text_tags):
    tags = numpy.unique(text_tags)

    tag_components = {}

    for tag in tags:
        tag_components[tag] = []

    for index, components in enumerate(text_components):
        tag = text_tags[index]
        tag_components[tag].append(components)

    tag_distributions = {}

    for tag in tags:
        means, variances = [], []

        for component in numpy.transpose(tag_components[tag]):
            means.append(numpy.mean(component))
            variances.append(numpy.var(component))

        tag_distributions[tag] = {'means': means, 'variances': variances}

    return tag_distributions


def predict(text_components, tag_distributions):
    predicted_tags = []

    for row in text_components:
        pdfs = {}

        for tag in list(tag_distributions.keys()):
            distribution = tag_distributions[tag]
            means, variances = distribution['means'], distribution['variances']
            pdfs[tag] = multivariate_normal.pdf(row, means, variances)

        predicted_tag = max(pdfs, key=pdfs.get)
        predicted_tags.append(predicted_tag)

    return predicted_tags


def run(lda, vectorizer, questions):
    train_texts, test_texts, train_tags, test_tags = train_test_split(
        questions.texts(), questions.tags(),
        test_size=(1 - TRAIN_PERCENTAGE)
    )

    train_text_vectors = vectorizer.fit_transform(train_texts)
    train_text_components = lda.fit_transform(train_text_vectors)

    print("Done training")

    tag_distributions = map_tag_distributions(
        train_text_components, train_tags
    )

    print("Done mapping distributions")

    test_text_vectors = vectorizer.transform(test_texts)
    test_text_components = lda.transform(test_text_vectors)

    print("Done transforming")

    predicted_tags = predict(test_text_components, tag_distributions)

    print("Done predicting")

    print(metrics.classification_report(test_tags, predicted_tags))


questions = Questions()

vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE, stop_words='english',
    analyzer="word", token_pattern=TOKEN_PATTERN,
    min_df=MIN_DF, max_features=MAX_FEATURES
)

lda = LatentDirichletAllocation(
    n_components=N_COMPONENTS, max_iter=MAX_ITER,
    learning_method='online', learning_offset=50,
    random_state=0
)

run(lda, vectorizer, questions)
