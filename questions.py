import csv


class Questions:
    PATH = "../language_questions.csv"

    def __init__(self, items=None):
        if items is None:
            with open(self.PATH, "r") as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
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
