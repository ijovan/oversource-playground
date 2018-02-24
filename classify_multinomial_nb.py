from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from questions import Questions
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Setting up the pipeline

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), stop_words='english',
    analyzer="word", min_df=1e-4,
    token_pattern="[><=!~:]{1,2}|[{};#]|[()//]{1,2}|[@A-Za-z_-]+"
)

classifier = MultinomialNB(alpha=1e-2)

pipeline = Pipeline([
    ('vectorizer', vectorizer), ('classifier', classifier)
])

# Loading the data set

questions = Questions()

train_texts, test_texts, train_tags, test_tags = train_test_split(
    questions.texts(), questions.tags(), test_size=0.35
)

# Training and prediction

pipeline.fit(train_texts, train_tags)

joblib.dump(pipeline, 'classifier.pkl')

predicted_tags = pipeline.predict(test_texts)

print(metrics.classification_report(test_tags, predicted_tags))
