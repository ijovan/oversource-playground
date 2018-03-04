import json
from questions import Questions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def languages_sentiment(questions, languages):
    for language in languages:
        texts = questions.for_language(language).texts()
        print("{0}: {1}".format(language, sentiment(texts)))


def sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()

    count = len(texts)
    compound, neutral, negative, positive = 0, 0, 0, 0

    for text in texts:
        score = analyzer.polarity_scores(text)
        compound += score['compound']
        neutral += score['neu']
        negative += score['neg']
        positive += score['pos']

    return {'compound': compound / count, 'neutral': neutral / count,
            'negative': negative / count, 'positive': positive / count}


languages = json.loads(open("languages.json", "r").read())
questions = Questions()

languages_sentiment(questions, languages)
