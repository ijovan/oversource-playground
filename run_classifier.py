from sklearn.externals import joblib

with open('example.txt', 'r') as myfile:
    text = myfile.read().replace('\n', ' ')

pipeline = joblib.load('classifier.pkl')

print(pipeline.predict([text]))
