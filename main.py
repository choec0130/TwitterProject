import tweepy
import json
import string
import vincent
import pandas as pd
import nltk
import re
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PIL
import requests
import urllib

from os import path
from PIL import Image
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk import bigrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter, defaultdict
from unidecode import unidecode


def get_tweets_of(screen_name):
    tweets = []
    # get 100 most recent tweets (max twitter allows)
    recent_tweets = api.user_timeline(screen_name=screen_name, count=100)

    tweets.extend(recent_tweets)

    oldest_tweet = recent_tweets[-1].id - 1

    # repeat until all tweets are acquired by getting 100 at a time
    while(len(recent_tweets) > 0):
        recent_tweets = api.user_timeline(screen_name=screen_name, count=100, max_id=oldest_tweet)
        tweets.extend(recent_tweets)
        oldest_tweet = tweets[-1].id-1

    tweet_strings = []
    for tweet in tweets:

        tweet_strings.extend([unidecode(tweet.text)])

    return tweet_strings

# file name is mytweets.json
fname = 'mytweets.json'

emoticons_str = r"""
     (?:
         [:=;] # Eyes
         [oO\-]? # Nose (optional)
         [D\)\]\(\]/\\OpP] # Mouth
     )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

# my API key and secret
consumer_key = ''
consumer_secret = ''

# my access token and secret
access_token = ''
access_secret = ''

# authorize
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)


def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):

    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def main():

    public_trump_tweets = api.search('Trump')

    for tweet in public_trump_tweets:
        print(tweet.text)
        analysis = TextBlob(tweet.text)
        print(analysis)

    # read my own Twitter timeline
    for status in tweepy.Cursor(api.home_timeline).items(10):
        print(status.text)
    for tweet in tweepy.Cursor(api.user_timeline).items():
        store(tweet._json)


    for tweet in public_trump_tweets:
        print(tweet.text)
        analysis = TextBlob(tweet.text)
        print(analysis)


def time_visualizatiton():
    with open(fname, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            # let's focus on hashtags only at the moment
            terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]
            # track when the hashtag is mentioned
            if '#kaplansba' in terms_hash:
                dates.append(tweet['created_at'])

    # a list of "1" to count the hashtags
    ones = [1] * len(dates)
    # the index of the series
    idx = pandas.DatetimeIndex(dates)
    # the actual series (at series of 1s for the moment)
    my_dates = pandas.Series(ones, index=idx)

    # series is resampled per minute
    per_minute = my_dates.resample('1Min', how='sum').fillna(0)

    time_chart = vincent.Line(my_dates)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json('time_chart.json')

    # all the data together
    match_data = dict(kaplansba=per_minute_i, amazonstudent=per_minute_s, hello=per_minute_e)
    # we need a DataFrame, to accommodate multiple series
    all_matches = pandas.DataFrame(data=match_data,
                                   index=per_minute_i.index)
    # Resampling as above
    all_matches = all_matches.resample('1Min', how='sum').fillna(0)

    # and now the plotting
    time_chart = vincent.Line(all_matches[['kaplansba', 'amazonstudent', 'hello']])
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.legend(title='Matches')
    time_chart.to_json('time_chart.json')

def co_occurrences():
    com = defaultdict(lambda: defaultdict(int))
    with open(fname, 'r') as f:
        # f is the file pointer to the JSON data set
        for line in f:
            tweet = json.loads(line)
            terms_only = [term for term in preprocess(tweet['text'])
                          if term not in stop
                          and not term.startswith(('#', '@'))]

            # Build co-occurrence matrix
            for i in range(len(terms_only) - 1):
                for j in range(i + 1, len(terms_only)):
                    w1, w2 = sorted([terms_only[i], terms_only[j]])
                    if w1 != w2:
                        com[w1][w2] += 1

    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    print(terms_max[:5])

    #  for specific term and its most frequent co-occurrences
    search_word = sys.argv[1]  # pass a term as a command-line argument
    count_search = Counter()
    for line in f:
        tweet = json.loads(line)
        terms_only = [term for term in preprocess(tweet['text'])
                      if term not in stop
                      and not term.startswith(('#', '@'))]
        if search_word in terms_only:
            count_search.update(terms_only)
    print("Co-occurrence for %s:" % search_word)
    print(count_search.most_common(20))

def parsing_tweets():

    with open(fname, 'r') as f:
        count_all = Counter()
        for line in f:
            tweet = json.loads(line)
            # create lists with various terms for parsing removal
            terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
            terms_all = [term for term in preprocess(tweet['text'])]
            terms_bigram = bigrams(terms_stop)

            # Count terms only once, equivalent to Document Frequency
            terms_single = set(terms_all)
            # Count hashtags only
            terms_hash = [term for term in preprocess(tweet['text'])
                          if term.startswith('#')]
            # Count terms only (no hashtags, no mentions)
            terms_only = [term for term in preprocess(tweet['text'])
                          if term not in stop and
                          not term.startswith(('#', '@'))]
            # mind the ((double brackets))
            # startswith() takes a tuple (not a list) if
            # we pass a list of inputs
            # Update the counter
            count_all.update(terms_stop)
        # Print the first 5 most frequent words
        print(count_all.most_common(5))

def visualize_data():
    word_freq = count_terms_only.most_common(20)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('term_freq.json')

    bar.to_json('term_freq.json', html_out=True, html_path='chart.html')

def store(tweet):
    print(json.dumps(tweet))

def geography_data(): #GeoJSON
    # Tweets are stored in "fname"
    with open(fname, 'r') as f:
        geo_data = {
            "type": "FeatureCollection",
            "features": []
        }
        for line in f:
            tweet = json.loads(line)
            if tweet['coordinates']:
                geo_json_feature = {
                    "type": "Feature",
                    "geometry": tweet['coordinates'],
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    }
                }
                geo_data['features'].append(geo_json_feature)

    # Save geo data
    with open('geo_data.json', 'w') as fout:
        fout.write(json.dumps(geo_data, indent=4))


if __name__ == "__main__":

    # main()

    my_tweets = get_tweets_of("chloeschoe")

    mask = np.array(
        Image.open(requests.get('http://theinspirationroom.com/daily/design/2012/6/new_twitter_logo.jpg', stream=True).raw))

    my_string = ""
    for tweet in my_tweets:
        my_string += tweet
    tweet_blob = TextBlob(my_string)
    my_polarity = tweet_blob.sentiment.polarity
    print("Polarity value of all tweets: " + str(my_polarity))
    if(my_polarity > 0):
        print("Tweets are generally positive!")
    else:
        print("Tweets are generally negative!")

    my_subjectivity = tweet_blob.sentiment.subjectivity
    print("Subjectivity value of all tweets: " + str(my_subjectivity))

    if(my_subjectivity >= 0.5):
        print("Tweets are generally subjective.")
    else:
        print("Tweets are generally objective.")

    stopwords = set(STOPWORDS)
    lemmatiser = WordNetLemmatizer()

    # get a list of standardized punctuation
    punctuation = list(string.punctuation)
    # stopwords = stopwords.words('english') + punctuation + ['rt', 'via', '...']

    # verbs_array = []
    # for word, tag in tweet_blob.tags:
    #     if tag == 'VB' and word in words.words():
    #         #  verbs_array.append(lemmatiser.lemmatize(word, pos="v"))
    #         verbs_array.append(word)
    #
    # verbs_file = open('tweets_verbs.txt', 'w')
    # for i in verbs_array:
    #     verbs_file.write("%s\n" % i)
    #
    # # Read the whole text.
    # verbs_text = open('tweets_verbs.txt').read()
    # stopwords = set(STOPWORDS)
    #
    # # display verb word cloud, tried verbs_text, final_verbs, verbs_file
    # wordcloud_verbs = WordCloud(max_font_size=50, stopwords=STOPWORDS, background_color='white',
    #                             mask=mask).generate(verbs_text)
    # wordcloud_verbs.to_file("verbs.png")
    # plt.imshow(wordcloud_verbs)
    # plt.axis("off")

    data = pd.DataFrame(my_tweets, columns=['tweets'])
    data['noun_phrases'] = data['tweets'].apply(lambda i: TextBlob(i).noun_phrases)
    nouns_array = []
    for i in range(len(data)):
        nouns_array.append('  '.join(data['noun_phrases'][i]))
    nouns_file = open('tweets_nouns.txt', 'w')
    for i in nouns_array:
        nouns_file.write("%s\n" % i)
    # Read the whole text.
    nouns_text = open('tweets_nouns.txt').read()
    stopwords = set(STOPWORDS)

    # display noun word cloud
    wordcloud_nouns = WordCloud(max_font_size=50, stopwords=STOPWORDS, background_color='white',
                                mask=mask).generate(nouns_text)
    wordcloud_nouns.to_file("nouns.png")
    plt.imshow(wordcloud_nouns)
    plt.axis("off")
