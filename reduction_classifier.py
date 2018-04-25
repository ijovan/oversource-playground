import numpy
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn import metrics
from regular_classifier import RegularClassifier


class ReductionClassifier:
    def __init__(self, reduction, model, vectorizer):
        self.reduction = reduction
        self.classifier = RegularClassifier(model)
        self.vectorizer = vectorizer

    def train_and_test(self, texts, tags, train_size):
        text_vectors = self.vectorizer.fit_transform(texts)
        coordinates = self.reduction.fit_transform(text_vectors)

        print("Done reducing")

        self.classifier.train_and_test(coordinates, tags, train_size)

        print("Done training")

    def show_topics(self, token_count=10):
        feature_names = self.vectorizer.get_feature_names()

        for index, topic in enumerate(self.reduction.components_):
            arguments = topic.argsort()[:-(token_count + 1):-1]
            tokens = [feature_names[i] for i in arguments]

            print('Topic #' + str(index))
            print(', '.join(tokens))
