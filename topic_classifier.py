import numpy
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn import metrics


class TopicClassifier:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def train(self, texts, tags):
        text_vectors = self.vectorizer.fit_transform(texts)
        text_components = self.model.fit_transform(text_vectors)

        self._map_tag_distributions(text_components, tags)

        print("Done training")

    def test(self, texts, tags):
        text_vectors = self.vectorizer.transform(texts)
        text_components = self.model.transform(text_vectors)

        predicted_tags = self._predict(text_components)

        print(metrics.classification_report(tags, predicted_tags))

    def train_and_test(self, texts, tags, train_size):
        train_texts, test_texts, train_tags, test_tags = train_test_split(
            texts, tags, test_size=(1 - train_size)
        )

        self.train(train_texts, train_tags)
        self.test(test_texts, test_tags)

    def show_topics(self, token_count=10):
        feature_names = self.vectorizer.get_feature_names()

        for index, topic in enumerate(self.model.components_):
            arguments = topic.argsort()[:-(token_count + 1):-1]
            tokens = [feature_names[i] for i in arguments]

            print('Topic #' + str(index))
            print(', '.join(tokens))

    def _map_tag_distributions(self, text_components, tags):
        tags_set = numpy.unique(tags)

        tag_components = {}

        for tag in tags_set:
            tag_components[tag] = []

        for index, components in enumerate(text_components):
            tag = tags[index]
            tag_components[tag].append(components)

        self.tag_distributions = {}

        for tag in tags_set:
            means, variances = [], []

            for component in numpy.transpose(tag_components[tag]):
                means.append(numpy.mean(component))
                variances.append(numpy.var(component))

            self.tag_distributions[tag] = \
                {'means': means, 'variances': variances}

    def _predict(self, text_components):
        predicted_tags = []

        for row in text_components:
            pdfs = {}

            for tag in list(self.tag_distributions.keys()):
                distribution = self.tag_distributions[tag]
                means, variances = \
                    distribution['means'], distribution['variances']
                pdfs[tag] = multivariate_normal.pdf(row, means, variances)

            predicted_tag = max(pdfs, key=pdfs.get)
            predicted_tags.append(predicted_tag)

        return predicted_tags
