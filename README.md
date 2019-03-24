# Shortest path similarity: item-based recommendation algorithm
## tl;dr
```
git clone https://github.com/grinya007/sp_i2i.git
cd sp_i2i
pip install -r requirements.txt
./recommend.py
```
## What it's all about?
Shortest path similarity is an alternative collaborative filtering item-based recommendation algorithm, it examines the order of happenings in the past to predict new happenings as opposed to conventional cosine similarity, which disregards the order in learning set. Code in this repository contains regular implementation of cosine similarity algorithm, POC implementation of shortest path similarity algorithm and a simple script ```./recommend.py``` that enables you to compare results of the two side by side. Recommendations here are based upon [MovieLens latest small](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) dataset. DISCLAIMER: recommendations of movies, based on user ratings, isn't the best application of shortest path similarity. Simply because the order of ratings is likely to have weak correlation with order of watching.
