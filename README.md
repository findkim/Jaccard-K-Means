Kim Ngo
Dong Wang
CSE40437 - Social Sensing
3 February 2016

Cluster tweets by utilizing the Jaccard Distance metric and K-means clustering algorithm

Usage: python k-means.py Tweets.json InitialSeeds.txt
Usage: python k-means++.py Tweets.json 25

K-Means Algorithm:
Utilizes dynamic programming to quickly reference jaccard distance between each pair.
Using the Jaccard Distance as a distance measurement for K-Means, there is a one-dimensional distance for each pair of tweets. Average Jaccard Distances are used to determine new clusters. For each tweet, an average distance is calculated for each cluster by dividing the sum Jaccard distance for each tweet in the cluster by the total number of tweets in that cluster (excludes the tweet in consideration). The tweet is then labeled with the cluster with the minimum average distance.

Determining Initial Seeds:
T
