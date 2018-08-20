import tweepy
import json
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from nltk.tokenize import word_tokenize
import nltk
import re
import operator
from collections import Counter


def main():
    nltk.download('punkt')

    #all keys and tokens are hidden for privacy
    consumer_key = ''  # API key
    consumer_secret = ''  # API secret key (hidden for privacy)
    access_token = ''
    access_secret = ''

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

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

    def tokenize(s):
        return tokens_re.findall(s)

    def preprocess(s, lowercase=False):
        tokens = tokenize(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens

    tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
    print(preprocess(tweet))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']




    # read my own Twitter timeline
    for status in tweepy.Cursor(api.home_timeline).items(10):
        print(status.text)
    for tweet in tweepy.Cursor(api.user_timeline).items():
        store(tweet._json)

#    with open('mytweets.json', 'r') as f:
#        line = f.readline()  # read only the first tweet/line
#        tweet = json.loads(line)  # load it as Python dict
#        print(json.dumps(tweet, indent=4))  # pretty-print


def store(tweet):
    print(json.dumps(tweet))


if __name__ == "__main__":
    main()
