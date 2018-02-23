from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from questions import Questions


# Setting up the pipeline

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), analyzer="word", stop_words='english'
)

classifier = SGDClassifier(
    verbose=True, random_state=42, max_iter=10, n_jobs=4
)

pipeline = Pipeline([('vect', vectorizer), ('clf', classifier)])

# Setting up the grid

grid_parameters = {
    'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
}

gscv = GridSearchCV(pipeline, grid_parameters, n_jobs=4, verbose=True)

# Loading the data set

questions = Questions()
questions.cut(20000)

# Training

gscv.fit(questions.texts(), questions.tags())

print(gscv.cv_results_)
print(gscv.best_score_)
print(gscv.best_params_)
