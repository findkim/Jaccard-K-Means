#!/usr/bin/env python

'''
Kim Ngo
Dong Wang
CSE40437 - Social Sensing
3 February 2016

Computes intial seeds given datafile for k-means clustering algorithm, k-means++
Reference: https://en.wikipedia.org/wiki/K-means%2B%2B
Cluster tweets by utilizing the Jaccard Distance metric and K-means clustering algorithm

Usage: k-means++.py [json file] [k clusters]
'''

import sys
import json
import re, string
import random
import bisect
from numpy import cumsum, random
import copy
from nltk.corpus import stopwords


regex = re.compile('[%s]' % re.escape(string.punctuation))
cachedStopWords = stopwords.words('english')

def accumu(l):
    total = 0
    for x in l:
        total += x
        yield total

class kMeans():
    def __init__(self, tweets, k):
        self.max_iterations = 1000
        self.tweets = tweets
        self.k = k
        self.seeds = self.initializeSeeds()

        self.clusters = {} # cluster to tweetID
        self.rev_clusters = {} # reverse index, tweetID to cluster
        self.jaccardMatrix = {} # stores pairwise jaccard distance in a matrix

        self.initializeClusters()
        self.initializeMatrix()

    def jaccardDistance(self, setA, setB):
        # Calcualtes the Jaccard Distance of two sets
        try:
            return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))
        except TypeError:
            print 'Invalid type. Type set expected.'

    def bagOfWords(self, string):
        # Returns a bag of words from a given string
        # Space delimited, removes punctuation, lowercase
        # Cleans text from url, stop words, tweet @, and 'rt'
        words = string.lower().strip().split(' ')
        for word in words:
            word = word.rstrip().lstrip()
            if not re.match(r'^https?:\/\/.*[\r\n]*', word) \
            and not re.match('^@.*', word) \
            and not re.match('\s', word) \
            and word not in cachedStopWords \
            and word != 'rt' \
            and word != '':
                yield regex.sub('', word)

    def initializeMatrix(self):
        # Dynamic Programming: creates matrix storing pairwise jaccard distances
        for ID1 in self.tweets:
            self.jaccardMatrix[ID1] = {}
            bag1 = set(self.bagOfWords(self.tweets[ID1]['text']))
            for ID2 in self.tweets:
                if ID2 not in self.jaccardMatrix:
                    self.jaccardMatrix[ID2] = {}
                bag2 = set(self.bagOfWords(self.tweets[ID2]['text']))
                distance = self.jaccardDistance(bag1, bag2)
                self.jaccardMatrix[ID1][ID2] = distance
                self.jaccardMatrix[ID2][ID1] = distance

    def initializeSeeds(self):
        # Computes initial seeds for k-means using k-means++ algorithm

        # 1. Choose one center uniformly at random from among the data points
        seed = random.choice(self.tweets.keys())        

        # 2. For each data point x, compute D(x),
        # the distance between x and the nearest center that has already been chosen
        seeds = set([seed])
        while len(seeds) < self.k:
            distanceMatrix = {}
            sum_sqr_dist = 0
            for seed in seeds:
                bag1 = set(self.bagOfWords(self.tweets[seed]['text']))
                for ID in self.tweets:
                    if ID == seed:
                        continue
                    bag2 = set(self.bagOfWords(self.tweets[ID]['text']))
                    dist = self.jaccardDistance(bag1, bag2)
                    if ID not in distanceMatrix or dist < distanceMatrix[ID]:
                        distanceMatrix[ID] = dist
            prob_dict = {}
            for ID in distanceMatrix:
                sum_sqr_dist += distanceMatrix[ID] * distanceMatrix[ID]
            for ID in distanceMatrix:
                prob_dict[ID] = distanceMatrix[ID] * distanceMatrix[ID] / sum_sqr_dist

            # 3. Choose one new data point at random as a new center,
            # using a weighted probability distribution
            # where a point x is chosen with probability proportional to D(x)^2.
            IDs, weights = prob_dict.keys(), prob_dict.values()
            seed = random.choice(IDs, p=weights)
            seeds.add(seed)
        
        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        return list(seeds)

    def initializeClusters(self):
        # Initialize tweets to no cluster
        for ID in self.tweets:
            self.rev_clusters[ID] = -1

        # Initialize clusters with seeds
        for k in range(self.k):
            self.clusters[k] = set([self.seeds[k]])
            self.rev_clusters[self.seeds[k]] = k

    def calcNewClusters(self):
        # Initialize new cluster
        new_clusters = {}
        new_rev_cluster = {}
        for k in range(self.k):
            new_clusters[k] = set()

        for ID in self.tweets:
            min_dist = float("inf")
            min_cluster = self.rev_clusters[ID]

            # Calculate min average distance to each cluster
            for k in self.clusters:
                dist = 0
                count = 0
                for ID2 in self.clusters[k]:
                    dist += self.jaccardMatrix[ID][ID2]
                    count += 1
                if count > 0:
                    avg_dist = dist/float(count)
                    if min_dist > avg_dist:
                        min_dist = avg_dist
                        min_cluster = k
            new_clusters[min_cluster].add(ID)
            new_rev_cluster[ID] = min_cluster
        return new_clusters, new_rev_cluster

    def converge(self):
        # Initialize previous cluster to compare changes with new clustering
        new_clusters, new_rev_clusters = self.calcNewClusters()
        self.clusters = copy.deepcopy(new_clusters)
        self.rev_clusters = copy.deepcopy(new_rev_clusters)

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < self.max_iterations:
            new_clusters, new_rev_clusters = self.calcNewClusters()
            iterations += 1
            if self.rev_clusters != new_rev_clusters:
                self.clusters = copy.deepcopy(new_clusters)
                self.rev_clusters = copy.deepcopy(new_rev_clusters)
            else:
                #print iterations
                return
    
    def printClusterText(self):
        # Prints text of clusters
        for k in self.clusters:
            for ID in self.clusters[k]:
                print self.tweets[ID]['text']
            print '\n'
 
    def printClusters(self):
        # Prints cluster ID and tweet IDs for that cluster
        for k in self.clusters:
            print str(k) + ':' + ','.join(map(str,self.clusters[k]))

    def printMatrix(self):
        # Prints jaccard distance matrix
        for ID in self.tweets:
            for ID2 in self.tweets:
                print ID, ID2, self.jaccardMatrix[ID][ID2]

    def printSeeds(self):
        for seed in self.seeds:
            print seed

def main():
    if len(sys.argv) != 3:
        print >> sys.stderr, 'Usage: %s [json file] [k clusters]' % (sys.argv[0])
        exit(-1)
    
    tweets = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets[tweet['id']] = tweet

    k = int(sys.argv[2])

    kmeans = kMeans(tweets, k)
    kmeans.converge()
    #kmeans.printClusterText()
    #kmeans.printSeeds()
    kmeans.printClusters()
    

if __name__ == '__main__':
    main()
