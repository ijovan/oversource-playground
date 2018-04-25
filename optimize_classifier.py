from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from stack_overflow import StackOverflow


vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
classifier = SGDClassifier(verbose=True, max_iter=10, n_jobs=4)
pipeline = Pipeline([('vect', vectorizer), ('clf', classifier)])

grid_parameters = {'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
grid = GridSearchCV(pipeline, grid_parameters, n_jobs=4, verbose=True)

stack_overflow = StackOverflow()
stack_overflow.cut(20000)

grid.fit(stack_overflow.texts(), stack_overflow.tags())

print(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
