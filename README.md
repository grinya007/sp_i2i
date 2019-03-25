# Shortest path similarity: item-based recommendation algorithm
## tl;dr
```
git clone https://github.com/grinya007/sp_i2i.git
cd sp_i2i
pip install -r requirements.txt
./recommend.py
```
## What it's all about?
Shortest path similarity is an alternative collaborative filtering item-based recommendation algorithm, it examines the order of happenings in the past to predict new happenings. As opposed to conventional [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which disregards the order in learning set. Code in this repository contains regular implementation of cosine similarity algorithm, proof-of-concept implementation of shortest path similarity algorithm and a simple script ```./recommend.py``` that enables you to compare results of the two side by side. Recommendations here are based upon [MovieLens latest small](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) dataset.

DISCLAIMER: recommendations of movies, based on user ratings, isn't the best application of shortest path similarity. Simply because the order of ratings is likely to have weak correlation with order of watching. Therefore, shortest path algorithm is predicting next movies to rate rather than next movies to watch. However, MovieLens dataset is the only, suitable for simple PoC, having understandable output.

## A bit of theory
There's quite a steady recipe for a recommendation system, established over the past decade. In short it reads as follows:
1. Try using [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
2. When the latter isn't good enough, give a try to [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)
3. When both of above struggle, go for [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning)

My suggestion is to set up paragraph 1a into the list. That is, there's an alternative to conventional collaborative filtering: Shortest path similarity.

First of all, it doesn't suffer the cold-start problem. That is, in situation when there's a new movie out and we know of ten people who rated the latter, also we know of twenty other movies rated by these ten people: our Cosine similarity algorithm has twenty movies to choose from for recommendations to the new one. Not much, really, especially given that ratings are not neccesserily equal, which means, these twenty movies are unlikely to be similar. Whilst, Shortest path will traverse this graph as deep as you want it to, and will likely have more to put on the table, and most of these nearest twenty movies may appear to be ranked below some more distant but more relevant ones.

Second of all, in applications, where _similarity_ is necessary but not sufficient for the _relevance_, Shortest path produces more logical recommendations. Many things, we do in our lives, have causal relationship. One wouldn't make himself a coffee when he had not been going to drink it, right? We wouldn't watch the third episode of True Detective after we had watched the first, but not the second. What does Cosine similarity algorithm know of these three episodes after examination of user's ratings? They are nearly equaly similar to each other. There's no clue to guess the order. Whilst, Shortest path does account for the order of happenings. Whenever you define user's ratings right (e.g. set up: watched entire episode: positive, stopped watching in the middle: negative, plus disregard stars, likes and others of the kind), Shortest path predicts most probable sequence with no sweat.

However, I don't mean to say there's something wrong with Cosine similarity algorithm itself. It works great when you match a document against learnt feature matrix to find similar documents (e.g. a search query against TFIDF matrix). Its common problem is misapplication, Cosine similarity isn't quite suitable when used as a standalone recommendation algorithm on, for example, videohostings and news websites, where similarity of articles doesn't neccessarily convert into high probability of click.

## A byte of theory
Consider the following graph of five movies, that are being evaluated by five users:
![Graph of five movies rated by five users](/img/5u5m_graph.png)

where each user gave a rating as shown in the table:

![Ratings table](/img/5u5m_table.png)

Let's apply Cosine similarity to given input. Actually, we only need the second figure. Let theta be the angle between vectors M1 and M2. To find cosine of that angle we use this formula:

![Cosine similarity formula](/img/cosine_i2i_formula_1.png)

derived from formula for [Euclidean dot product](https://en.wikipedia.org/wiki/Euclidean_vector#Dot_product). Note, that n = 3 in this case, we can't take into account ratings, given by U2 and U3, as their rating for M1 in unknown. What we get is:

![M1 to M2 similarity](/img/cosine_i2i_formula_m1_m2_1.png)

Wow, looks like they're very similar, although, everything is relative. Let's have a look at all similarities:

![Cosine similarity recommendations](/img/cosine_i2i_recs.png)

Look, M5 happens to be even more similar. Note, that in real application it's always better to normilize input values (e.g. make them always remain between 0 and 1). In this particular case normalization wouldn't make any difference in ranking, but when a difference between values of similarity is getting closer to precision limitation of floating point number, normalization comes in handy.

Ok, so now we have some results. But not all 5 movies have gotten 4 recommendations, even though sparsity of the input matrix isn't that high.

