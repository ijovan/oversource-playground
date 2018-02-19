import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import pandas as pd


def dateparse(stamp):
    return pd.datetime.fromtimestamp(float(stamp)).date()


def time_series():
    rcParams['figure.figsize'] = 15, 6

    questions = pd.read_csv('../ts.csv', delimiter=',', parse_dates=True,
                            date_parser=dateparse, index_col='created_at',
                            names=['created_at', 'comment_count'],
                            header=None)

    questions = questions['2013':'2017'].sort_index()
    questions = \
        questions.groupby(lambda x: x)['comment_count'].agg(['sum'])

    ts = questions['sum']

    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    plt.plot(ts, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.show()


time_series()
