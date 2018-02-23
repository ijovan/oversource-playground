from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from questions import Questions


# Setting up the pipeline

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), analyzer="word", stop_words='english',
    min_df=1e-4,
    token_pattern="[><=!~:\-\+]{1,3}|[{};#$@]|[()/]{1,2}|[A-Za-z]+"
)

classifier = MultinomialNB(alpha=1e-2)

pipeline = Pipeline([('vect', vectorizer), ('clf', classifier)])

# Setting up the grid

grid_parameters = {
    'vect__use_idf': [True, False]
}

gscv = GridSearchCV(pipeline, grid_parameters, n_jobs=1, verbose=True)

# Loading the data set

questions = Questions()
questions.cut(30000)

# Training

gscv.fit(questions.texts(), questions.tags())

print(gscv.cv_results_)
print(gscv.best_score_)
print(gscv.best_params_)
