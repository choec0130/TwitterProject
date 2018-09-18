import tweepy
import json
import sys
import string
import vincent
import pandas
import nltk
import re
import operator
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
from collections import Counter
from collections import defaultdict

# import from nltk
nltk.download('stopwords')
nltk.download('punkt')

# get a list of standardized punctuation
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '...']
# file name is mytweets.json
fname = 'mytweets.json'

dates = []

# preprocess to recognize Twitter-like strings
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

def process_previous_tweets():
    with open('mytweets.json', 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'])

#def request_to_api():
 #   GET https://api.twitter.com/1.1/followers/list.json?cursor=-1&screen_name=twitterdev&skip_status=true&include_user_entities=false
#def giveaways():


def check_tweet():
    giveaway_terms = ['giveaway', 'raffle', 'free stuff', 'sweepstakes']
    if any(x in giveaway_terms for x in tweet):
        if("retweet" or "rt" in tweet):
            # retweet the tweet
        else if("like" in tweet):
            # like the tweet
        else if("tag" in tweet):
            # tag friend in tweet

def main():

    # my API key and secret
    consumer_key = 'o2HBhet9G40bo2KtFkwhMimf2'
    consumer_secret = 'DUfJa7yxdHsY0QMBJpjviy6t849yV5TGF27rFvwsqZCO8x7sMr'  # API secret key

    # my access token and secret
    access_token = '1138010744-A6vcezFNOkxDGW74DlnZHe3S1SpjoNZEB13F0tm'
    access_secret = 'kQlIaoX4reBH5b7KHVXMDU1dnFcOujM6ayxtOBFnXbBEg'

    # authorize
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    tweet = 'RT @marcobonzanini: this is an eexample! :D http://example.com #NLP'
    print(preprocess(tweet))
    # ['RT', '@marcobonzanini', ':', 'this', 'is', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']

    # read my own Twitter timeline
    for status in tweepy.Cursor(api.home_timeline).items(10):
        print(status.text)
    for tweet in tweepy.Cursor(api.user_timeline).items():
        store(tweet._json)

#   put all tweets into mytweets.json

#    with open('mytweets.json', 'r') as f:
#        line = f.readline()  # read only the first tweet/line
#        tweet = json.loads(line)  # load it as Python dict
#        print(json.dumps(tweet, indent=4))  # pretty-print

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

def sentiment_term_probabilities():
    # n_docs is the total n. of tweets
    p_t = {}
    p_t_com = defaultdict(lambda: defaultdict(int))

    for term, n in count_stop_single.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    positive_words = []
    negative_words = []

def sentiment_calculate_pmi():
    pmi = defaultdict(lambda: defaultdict(int))
    for t1 in p_t:
        for t2 in com[t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)

    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc

    semantic_sorted = sorted(semantic_orientation.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]

    print(top_pos)
    print(top_neg)
    print("ITA v WAL: %f" % semantic_orientation['#itavwal'])
    print("SCO v IRE: %f" % semantic_orientation['#scovire'])
    print("ENG v FRA: %f" % semantic_orientation['#engvfra'])
    print("#ITA: %f" % semantic_orientation['#ita'])
    print("#FRA: %f" % semantic_orientation['#fra'])
    print("#SCO: %f" % semantic_orientation['#sco'])
    print("#ENG: %f" % semantic_orientation['#eng'])
    print("#WAL: %f" % semantic_orientation['#wal'])
    print("#IRE: %f" % semantic_orientation['#ire'])

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
    main()
