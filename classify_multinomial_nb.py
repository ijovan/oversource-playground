from questions import Questions
from regular_classifier import RegularClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


TRAIN_SIZE = 0.65

NGRAM_RANGE = (1, 2)
MIN_DF = 1e-4
TOKEN_PATTERN = \
    '[><=!~:]{1,2}|[{};#]|[()//]{1,2}|[@A-Za-z_-]+'

ALPHA = 1e-2


vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE, stop_words='english',
    analyzer='word', min_df=MIN_DF, token_pattern=TOKEN_PATTERN
)

model = MultinomialNB(alpha=1e-2)

questions = Questions()

classifier = RegularClassifier(model, vectorizer)
classifier.train_and_test(questions.texts(), questions.tags(), TRAIN_SIZE)
