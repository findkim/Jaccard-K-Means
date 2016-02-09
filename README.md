Kim Ngo

Dong Wang

CSE40437 - Social Sensing

3 February 2016

## Files
k-means.py: given an inital list of seeds, the program clusters tweets by utilizing the Jaccard Distance metric and K-means clustering algorithm

k-means++.py: without an initial list of seeds, the program computes an initial list of seeds with the k-means++ algorithm and clusters tweets by utilizing the Jaccard Distance metric and K-means clustering algorithm

## Usage
`python k-means.py Tweets.json InitialSeeds.txt`
`python k-means++.py Tweets.json 25`

## K-Means Algorithm
Utilizes dynamic programming to quickly reference jaccard distance between each pair.
Using the Jaccard Distance as a distance measurement for K-Means, there is a one-dimensional distance for each pair of tweets. Average Jaccard Distances are used to determine new clusters. For each tweet, an average distance is calculated for each cluster by dividing the sum Jaccard distance for each tweet in the cluster by the total number of tweets in that cluster (excludes the tweet in consideration). The tweet is then labeled with the cluster with the minimum average distance.

## Determining Initial Seeds
I implemented the k-means++ algorithm proposed by David Arthur and Sergei Vassilvitskii. The first seed is randomly selected from the set of tweets. Then, for each tweet t, the distance between t and the nearest seed that has already been chosen is computed. Another seed is then chosen with probability of its distance squared among the sum of all distances squared. These steps are computed until k seeds have been selected.

for each t, D(t)^2 / sum(D(t)^2), where D(t) is the distance between t and the nearest seed.

A dictionary is used to record the shortest Jaccard distance between a tweet and the neartest seed. Once all of the nearest distance is computed, I sum the squares of the distances and compute the probability of each tweet by dividing the squared distance by the sum of squares and storing it into another dictionary--key: tweetID, value: probability. 

With a dictionary of probabilities, I randomly choose with weighted probability the next seed to be added. This is done by splitting the dictionary to a list of keys and list of corresponding values, which are the corresponding probabilities and using `numpy.random.choice` to select at random with weighted probability.


## References
[https://en.wikipedia.org/wiki/K-means_clustering]

[https://en.wikipedia.org/wiki/K-means%2B%2B]
