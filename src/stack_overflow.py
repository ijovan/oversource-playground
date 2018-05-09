import csv
import codecs
import random


class StackOverflow:
    PATH = "stack_overflow.csv"

    def __init__(self, items=None):
        if items is None:
            reader = csv.reader(codecs.open(self.PATH, 'rU', 'utf-8'))
            self.items = [row for row in reader]
        else:
            self.items = items

    def for_language(self, language):
        filtered = list(filter(lambda question:
                               question[2] == language, self.items))
        return StackOverflow(filtered)

    def tags(self):
        return list(map(lambda question: question[2], self.items))

    def texts(self):
        return list(map(lambda question: question[3], self.items))

    def pairs(self):
        return list(map(lambda question:
            [question[2], question[3]], self.items
        ))

    def shuffle(self):
        random.shuffle(self.items)

    def cut(self, limit):
        self.items = self.items[:limit]
