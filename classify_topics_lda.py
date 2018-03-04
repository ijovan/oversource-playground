from questions import Questions
from topic_classifier import TopicClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_SIZE = 0.8

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 3000
MIN_DF = 1e-5
TOKEN_PATTERN = \
    'c[#\+]{1,2}|[!~][=]{1,2}|[:\-\+]{2}|[{}$@]|[()/]{2}|[A-Za-z]+'

N_COMPONENTS = 15
MAX_ITER = 5


questions = Questions()

vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE, stop_words='english',
    analyzer='word', token_pattern=TOKEN_PATTERN,
    min_df=MIN_DF, max_features=MAX_FEATURES
)

model = LatentDirichletAllocation(
    n_components=N_COMPONENTS, max_iter=MAX_ITER,
    learning_method='online', learning_offset=50,
    random_state=0
)

classifier = TopicClassifier(model, vectorizer)
classifier.train_and_test(questions.texts(), questions.tags(), TRAIN_SIZE)
classifier.show_topics()
