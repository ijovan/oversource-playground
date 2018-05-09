from src.stack_overflow import StackOverflow
from src.reduction_classifier import ReductionClassifier
from sklearn.decomposition import NMF
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_SIZE = 0.8

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 2000
MIN_DF = 1e-4
TOKEN_PATTERN = \
    'c[#\+]{1,2}|[!~][=]{1,2}|[:\-\+]{2}|[{}$@]|[()/]{2}|[A-Za-z]+'

N_COMPONENTS = 40
MAX_ITER = 500
BETA_LOSS = 'frobenius'
SOLVER = 'mu'
TOL = 1e-4
ALPHA = .1
L1_RATIO = .05


stack_overflow = StackOverflow()

vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE, stop_words='english',
    analyzer='word', token_pattern=TOKEN_PATTERN,
    min_df=MIN_DF, max_features=MAX_FEATURES
)

# reduction = LatentDirichletAllocation(n_components=15)
reduction = NMF(
    n_components=N_COMPONENTS, random_state=1,
    beta_loss=BETA_LOSS, solver=SOLVER, tol=TOL,
    max_iter=MAX_ITER, alpha=ALPHA, l1_ratio=L1_RATIO
)

# model = KNeighborsClassifier()
# model = MultinomialNB()
model = SGDClassifier(n_jobs=4)

classifier = ReductionClassifier(reduction, model, vectorizer)
classifier.train_and_test(
    stack_overflow.texts(), stack_overflow.tags(), TRAIN_SIZE
)
classifier.show_topics()
