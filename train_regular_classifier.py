from src.stack_overflow import StackOverflow
from src.regular_classifier import RegularClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


TRAIN_SIZE = 0.65

NGRAM_RANGE = (1, 2)
MIN_DF = 1e-4
TOKEN_PATTERN = \
    '[><=!~:\-\+]{1,3}|[{};#$@]|[()/]{1,2}|[A-Za-z]+'

LOSS = 'hinge'
PENALTY = 'l2'
MAX_ITER = 10
ALPHA = 1e-5


vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE, stop_words='english',
    analyzer='word', min_df=MIN_DF, token_pattern=TOKEN_PATTERN
)

model = SGDClassifier(
    loss=LOSS, penalty=PENALTY, max_iter=MAX_ITER, n_jobs=4, alpha=ALPHA
)
# model = MultinomialNB(alpha=1e-2)

stack_overflow = StackOverflow()

classifier = RegularClassifier(model, vectorizer)
classifier.train_and_test(stack_overflow.texts(), stack_overflow.tags(), TRAIN_SIZE)
classifier.save()
