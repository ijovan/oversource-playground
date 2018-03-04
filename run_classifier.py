from regular_classifier import RegularClassifier

with open('example.txt', 'r') as myfile:
    text = myfile.read().replace('\n', ' ')

classifier = RegularClassifier()
classifier.load()

print(classifier.predict([text])[0])
