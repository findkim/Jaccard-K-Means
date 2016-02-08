#!/usr/bin/env python

'''
Kim Ngo
Dong Wang
CSE40437 - Social Sensing
3 February 2016

Cluster tweets by utilizing the Jaccard Distance metric and K-means clustering algorithm
'''

import sys
import json
import copy
import re, string

regex = re.compile('[%s]' % re.escape(string.punctuation))

class kMeans():
    def __init__(self, seeds, tweets):
        self.seeds = seeds
        self.tweets = tweets
        self.max_iterations = 1000
        self.k = len(seeds)

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
        return set(regex.sub('',string.lower().strip()).split(' '))

    def initializeMatrix(self):
        # Dynamic Programming: creates matrix storing pairwise jaccard distances
        for ID1 in self.tweets:
            self.jaccardMatrix[ID1] = {}
            bag1 = self.bagOfWords(self.tweets[ID1]['text'])
            for ID2 in self.tweets:
                if ID2 not in self.jaccardMatrix:
                    self.jaccardMatrix[ID2] = {}
                bag2 = self.bagOfWords(self.tweets[ID2]['text'])
                distance = self.jaccardDistance(bag1, bag2)
                self.jaccardMatrix[ID1][ID2] = distance
                self.jaccardMatrix[ID2][ID1] = distance

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
            min_dist = 1
            min_cluster = self.rev_clusters[ID]

            # Calculate min average distance to each cluster
            for k in self.clusters:
                dist = 0
                count = 0
                for ID2 in self.clusters[k]:
                    if ID != ID2:
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
        old_clusters, old_rev_cluster = self.calcNewClusters()
        self.clusters = old_clusters
        self.rev_clusters = old_rev_cluster

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < self.max_iterations:
            new_clusters, new_rev_cluster = self.calcNewClusters()
            iterations += 1
            if old_rev_cluster == new_rev_cluster:
                break
            else:
                old_clusters = self.clusters
                old_rev_cluster = self.rev_clusters
                self.clusters = new_clusters
                self.rev_clusters = new_rev_cluster
        #print iterations
    
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

def main():
    if len(sys.argv) != 3:
        print >> sys.stderr, 'Usage: %s [json file] [seeds file]' % (sys.argv[0])
        exit(-1)
    
    tweets = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets[tweet['id']] = tweet
    
    f = open(sys.argv[2])
    seeds = [int(line.rstrip(',\n')) for line in f.readlines()]
    f.close()

    kmeans = kMeans(seeds, tweets)
    kmeans.converge()
    #kmeans.printClusterText()
    kmeans.printClusters()
    

if __name__ == '__main__':
    main()
