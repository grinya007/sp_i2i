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

First of all, it doesn't suffer the cold-start problem. That is, in situation when there's a new movie out and we know of ten people who rated the latter, also we know of twenty other movies rated by these ten people: our Cosine similarity algorithm has twenty movies to choose from for recommendations to the new one. Not much, really, especially given that ratings are not necessarily equal, which means, these twenty movies are unlikely to be similar. Whilst, Shortest path will traverse this graph as deep as you want it to, and will likely have more to put on the table, and most of these nearest twenty movies may appear to be ranked below some more distant but more relevant ones.

Second of all, in applications, where _similarity_ is necessary but not sufficient for the _relevance_, Shortest path produces more logical recommendations. Many things we do in our lives, have causal relationship. One wouldn't make himself a coffee when he had not been going to drink it, right? We wouldn't watch the third episode of True Detective after we had watched the first, but not the second. What does Cosine similarity algorithm know of these three episodes after examination of user's ratings? They are nearly equally similar to each other. There's no clue to guess the order. Whilst, Shortest path does account for the order of happenings. Whenever you define user's ratings right (e.g. set up: watched entire episode: positive, stopped watching in the middle: negative, plus disregard stars, likes and others of the kind), Shortest path predicts most probable sequence with no sweat.

However, I don't mean to say there's something wrong with Cosine similarity algorithm itself. It works great when you match a document against learnt feature matrix to find similar documents (e.g. a search query against TFIDF matrix). Its common problem is misapplication, Cosine similarity isn't quite suitable when used as a standalone recommendation algorithm on, for example, videohostings and news websites, where similarity of articles doesn't necessarily convert into high probability of click.

## A byte of theory
Consider the following graph of five movies, that are being evaluated by five users:
![Graph of five movies rated by five users](/img/5u5m_graph.png)

where each user gave a rating as shown in the table:

![Ratings table](/img/5u5m_table.png)

Let's apply Cosine similarity to given input. Actually, we only need the second figure. Let theta be the angle between vectors M1 and M2. To find cosine of that angle we use this formula:

![Cosine similarity formula](/img/cosine_i2i_formula_1.png)

derived from formula for [Euclidean dot product](https://en.wikipedia.org/wiki/Euclidean_vector#Dot_product). Note that n = 3 in this case, we can't take into account ratings, given by U2 and U3, as their rating for M1 in unknown. What we get is:

![M1 to M2 similarity](/img/cosine_i2i_formula_m1_m2_1.png)

Wow, looks like they're very similar, although, everything is relative. Let's have a look at all similarities:

![Cosine similarity recommendations](/img/cosine_i2i_recs.png)

Look, M5 happens to be even more similar. Note that in real application it's always better to normalize input values (i.e. scale them proportionally). In this particular case normalization wouldn't make any difference in ranking, but when a difference between values of similarity is getting closer to precision limitation of floating point number, normalization comes in handy.

Ok, so now we have some results. But not all 5 movies have gotten 4 recommendations, even though sparsity of the input matrix isn't that high. This is what's usually called the cold-start problem, also known as sparsity problem. Cosine similarity demands more data not only to produce _better_ recommendations, but to produce recommendations at all.

There's another concern, which isn't that evident as the former. Similarities are mirrored. Meaning that, for instance, cos(M1-M2) = cos(M2-M1). Well, sounds logical, as this is _similarity_. But we're trying to make up recommendations in the first place. Imagine that M1 is "Back to the Future" and M2 is "Back to the Future II". While they are equally similar in both ways, are they equally relevant as recommendations to each other? Well, you might argue that of course, Marty gets back to 1955 right in the beginning of the second episode. But let me kindly ask you to scroll a little bit up, look at the first figure and follow all arrows between M1 and M2. Right, two users have chosen to go in the same direction and none went the other way around. What if this is a valid indication of inequality of the relevance? Use deep learning to sort it out would be a good answer. But there's a simpler way.

Let's see if Shortest path similarity algorithm can do any good in such situation. First of all let's flatten the original graph into a table of user paths, for clarity:

![User paths](/img/sp_user_paths.png)

Sweet, now, since we have a graph, we can assign different lengths to its edges. In this particular case, the most logical would be to derive lengths from movie ratings. I came up with the following formula for that:

![Length formula](/img/sp_edge_length_formula.png)

where v and v' are connected vertices, r_iv and r_iv' are ratings, given by i-th user, w_r_i be the weight of i-th user's opinion and w_v' be the global weight of given recommendation candidate. However, when you would want to try Shortest path similarity on some other data, I encourage you to think thoroughly of what is the best way to compute lengths. My proposed formula isn't crucial for Shortest path similarity algorithm. Cosine distance could be a great option here, by the way.

For simplicity, in this case with 5 movies and 5 users, I assume that all users have equal weights and all movies have equal weights. Thereby, I ended up with the following adjacency list:

![Adjacency list](/img/sp_movies_adjacency_list.png)

Note that this isn't a list of recommendations yet, although, each nearest vertice will inevitably become the most relevan recommendation. Now, when it's all said and done, it's time to look for shortest path. I propose to use breadth-first search with priority queue, which, for M1, will work as follows:

![Shortest path for M1](/img/sp_recs_for_m1.png)

From M1 it will guide us towards M2, then to M4 through M2, since there's no direct flight, then to M5 through M2 and after all to M1 through M2 and M4. After entering the graph through each of its vertices we end up with the following table of recommendations:

![Shortest path recommendations](/img/sp_recs.png)

Note that now we have full recommendations, we can't get more out of five movies. Although, I wouldn't say this is a great achievement here, since we allowed a single user to connect some of vertices, which isn't necessarily a good idea in real application. But what's more important, now, even though M2 is the most relevant recommendation for M1, M1, in turn, is the least relevant to M2.

## Conclusion
There's no proof of Fermat's Theorem in my proposed Shortest path similarity algorithm. All pieces of it are superficial and broadly used in software development. Although, not only that it seems to be very natural as a part of recommendation system, but I have proven results of it, outperforming other conventional algorithms on production. If you're interested in greater details or have a dataset or application in which you'd like to try Shortest path similarity: please, don't hesitate to reach out.
