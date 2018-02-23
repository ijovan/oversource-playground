from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from questions import Questions
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Setting up the pipeline

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), stop_words='english', analyzer="word", min_df=1e-4,
    token_pattern="[><=!~:\-\+]{1,3}|[{};#$@]|[()/]{1,2}|[A-Za-z]+"
)

classifier = SGDClassifier(
    loss='hinge', penalty='l2', verbose=True, random_state=42,
    max_iter=10, n_jobs=4, alpha=1e-5
)

pipeline = Pipeline([
    ('vectorizer', vectorizer), ('classifier', classifier)
])

# Loading the data set

questions = Questions()
questions.cut(100000)

train_texts, test_texts, train_tags, test_tags = train_test_split(
    questions.texts(), questions.tags(), test_size=0.35
)

# Training and prediction

pipeline.fit(train_texts, train_tags)

predicted_tags = pipeline.predict(test_texts)

print(metrics.classification_report(test_tags, predicted_tags))
