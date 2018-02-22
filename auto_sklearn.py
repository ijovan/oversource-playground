import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from questions import Questions
from sklearn.feature_extraction.text import CountVectorizer


def classify(texts, tags):
    train_texts, test_texts, train_tags, test_tags = \
            train_test_split(texts, tags, test_size=0.8)

    train_texts = CountVectorizer().fit_transform(train_texts)

    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(train_texts, train_tags)
    y_hat = automl.predict(test_texts)

    print("Accuracy score: ", sklearn.metrics.accuracy_score(test_tags, y_hat))
    print("Models used: ", automl.get_models_with_weights())


questions = Questions()
questions.shuffle()
questions.cut(1000)

classify(questions.texts(), questions.tags())
