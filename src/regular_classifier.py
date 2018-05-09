from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics


class RegularClassifier:
    SAVE_PATH = 'classifier.pkl'

    def __init__(self, model=None, vectorizer=None):
        if model and vectorizer:
            self.pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', model)
            ])
        elif model:
            self.pipeline = Pipeline([('classifier', model)])

    def train(self, texts, tags):
        self.pipeline.fit(texts, tags)

    def test(self, texts, tags):
        predicted_tags = self.predict(texts)

        print(metrics.classification_report(tags, predicted_tags))

    def train_and_test(self, texts, tags, train_size):
        train_texts, test_texts, train_tags, test_tags = train_test_split(
            texts, tags, test_size=(1 - train_size)
        )

        self.train(train_texts, train_tags)
        self.test(test_texts, test_tags)

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def save(self, path=SAVE_PATH):
        joblib.dump(self.pipeline, 'classifier.pkl')

    def load(self, path=SAVE_PATH):
        self.pipeline = joblib.load('classifier.pkl')
