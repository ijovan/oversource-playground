import csv
import codecs
import random


class HackerNews:
    PATH = "hacker_news.csv"

    def __init__(self, items=None):
        if items is None:
            reader = csv.reader(codecs.open(self.PATH, 'rU', 'utf-8'))
            self.items = [row for row in reader]
        else:
            self.items = items

    def texts(self):
        return list(map(lambda text: text[1], self.items))

    def shuffle(self):
        random.shuffle(self.items)

    def cut(self, limit):
        self.items = self.items[:limit]
