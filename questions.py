import csv
import codecs
import random


class Questions:
    PATH = "questions.csv"

    def __init__(self, items=None):
        if items is None:
            reader = csv.reader(codecs.open(self.PATH, 'rU', 'utf-8'))
            self.items = [row for row in reader]
        else:
            self.items = items

    def for_language(self, language):
        filtered = list(filter(lambda question:
                               question[2] == language, self.items))
        return Questions(filtered)

    def texts(self):
        return list(map(lambda question: question[3], self.items))

    def tags(self):
        return list(map(lambda question: question[2], self.items))

    def shuffle(self):
        random.shuffle(self.items)

    def cut(self, limit):
        self.items = self.items[:limit]
