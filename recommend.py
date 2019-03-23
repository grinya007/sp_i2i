#!/usr/bin/env python3
import pandas as pd
from search import Search
from algs import CosineSimilarity, ShortestPath
from time import time
import os

# LIMIT applies to everything, shown on the screen:
# movies search results, movies recommendations
LIMIT=5

script_dir = os.path.dirname(os.path.realpath(__file__))
movies_csv = '{}/data/movies.csv'.format(script_dir)
ratings_csv = '{}/data/ratings.csv'.format(script_dir)

pd.set_option('display.max_colwidth', 60)

# converter function, truncates long movie titles
def truncate_title(title):
    if len(title) > 50:
        title = title[:40] + ' ... ' + title[-6:]
    return title

print('Loading...')

# loading smart searcher, two instances of which:
# one is matching words, existing in movie titles
movies = pd.read_csv(movies_csv, index_col = 'movieId', converters = {'title': truncate_title})
word_searcher = Search(df = movies, column = 'title', analyzer = 'word', ngram_range = (1,1))

# and the other one is matching ngrams against features of the former
words = pd.DataFrame(word_searcher.features, columns = ['feat'])
char_searcher = Search(df = words, column = 'feat', analyzer = 'char', ngram_range = (3,3))
# please see search.py for details of implementation

# loading movie ratings
ratings = pd.read_csv(ratings_csv, index_col = ['userId', 'movieId'])

t = time()
print('Cosine similarity recommendations are calculated in   ', end='', flush=True)
# doing conventional i2i by means of cosine similarity algorithm (please see algs.py)
cosine = CosineSimilarity(ratings, limit=LIMIT)
print('{:.3f} s'.format(time() - t))

t = time()
print('Shortest path recommendations are calculated in       ', end='', flush=True)
# doing shortest path i2i (please see algs.py)
shortp = ShortestPath(ratings, limit=LIMIT)
print('{:.3f} s'.format(time() - t))
print("\n")

# renders recommendations
def show_recs_for(movie_id):
    print('===> {}'.format(movies.loc[movie_id]['title']))
    
    cosine_recs = cosine.recommend(movie_id)
    cosine_recs = cosine_recs.join(movies, on='movieId', how='inner')
    cosine_recs = cosine_recs.reset_index('movieId')
    
    shortp_recs = shortp.recommend(movie_id)
    shortp_recs = shortp_recs.join(movies, on='movieId', how='inner')
    shortp_recs = shortp_recs.reset_index('movieId')
    recs = cosine_recs.join(shortp_recs, lsuffix='_c', rsuffix='_s')[['title_c', 'title_s']]
    recs = recs.rename(columns = {'title_c': 'Cosine similarity alg:', 'title_s': 'Shortest path alg:'})
    print(recs)

# search and recommend forever
while True:
    r = input('Type in movie title (q to quit): ')
    if not r:
        continue
    if r == 'q':
        break
    # if less than two words match features of word searcher
    # we're going for ngram search first, compulsory T9, if you will
    if len(list(filter(lambda w: w in word_searcher.features, r.split(' ')))) < 2:
        r = char_searcher.search(r)
        r = ' '.join(r.head(n=4)['feat'].values)
        if not r:
            continue
        print("looks like you're searching for: " + r)
    r = word_searcher.search(r)
    r = r.reset_index(level = 'movieId')
    print("\n")
    if r.shape[0] == 1:
        show_recs_for(r['movieId'].values[0])
    elif r.shape[0] > 1:
        print(r.head(n=LIMIT)['title'].to_frame())
        print("\n")
        index = input('Which one? (0-{}, default 0): '.format(min(r.shape[0], LIMIT)-1))
        print("\n")
        if not index:
            index = '0'
        if index.isdigit() and int(index) < min(r.shape[0], LIMIT):
            show_recs_for(r['movieId'].values[int(index)])
    print("\n")

