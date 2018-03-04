from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from questions import Questions


vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
classifier = MultinomialNB(alpha=1e-2)
pipeline = Pipeline([('vect', vectorizer), ('clf', classifier)])

grid_parameters = {'vect__use_idf': [True, False]}
grid = GridSearchCV(pipeline, grid_parameters, n_jobs=4, verbose=True)

questions = Questions()
questions.cut(30000)

grid.fit(questions.texts(), questions.tags())

print(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
