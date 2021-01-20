

#choose n*p(p is a parameter) for the training set:
import csv

import numpy as np
import pandas as pd
import random

from ID3 import Node, fit, getAttributeCalumn, getMajorityClass

#todo: check
def get_euclidean_dist(vec1,vec2):
    sum = 0
    for i,j in zip(vec1,vec2):
        sum += pow(i-j,2)
    return np.math.sqrt(sum)


#todo : normalize the centroid so that the distance in the classification will be from all the vectore
def get_centroid(df,random_data): #the centroid will have only the features in its vector
    pd.DataFrame(random_data).to_csv("C:/My Stuff/studies/2021a/AI/hw3/random_data.csv")
    number_of_features = len(df[0])
    len_of_random_data = len(random_data)
    feature_average_array = []
    for feature_place in range(1, number_of_features):  # for each feature #the first feature is diagnostic so i give it out
        sum_for_feature = 0
        for i in random_data:
            feature_value_in_line_i = i[feature_place]
            sum_for_feature += feature_value_in_line_i
        average = sum_for_feature/len_of_random_data
        feature_average_array.append(average)  # without the first feature which is diagnostic,is it okay?
    return feature_average_array


def getClassification(node,line_to_classify,header):
    if node.classification is not None:
        return node.classification
    else:
        # there is partition here
        attribute_column = getAttributeCalumn(header, node)
        if line_to_classify[attribute_column] < node.partition_feature_and_limit[1]:
            return getClassification(header, node.left, line_to_classify)  # under limit
        else:
            return getClassification(header, node.right,line_to_classify)  # above or equal to limit


def KNN(p, number_of_trees_N, k):
    true = 0
    false = 0
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    n = len(data_without_header)
    tree_array = []
    #build N trees with ID3 - each tree will get p*n examples, from the n examples in the training set
    for i in range(0, number_of_trees_N):
        number_of_random = round(p*n)
        random_data = random.sample(list(data_without_header), number_of_random)
        df = (header, random_data)
        centroid = get_centroid(df,random_data)
        node = Node()
        fit(df, node) #fit = algorithm ID3 thet used in ID3.py  #no pruning todo: maybe change to with pruning
        tree_array.append((node, centroid))

    #check the classification of examples in test.csv:
    df_test = pd.read_csv("test.csv", header=0)
    test_data_without_header = df_test.to_numpy()

    df_test = (header, test_data_without_header)
    tree_dist_arr = []
    for line_to_classify in test_data_without_header:
        line_to_classify.pop(0) #todo: test it removes the first element
        for tree, centroid in tree_array:
            euclidean_dist = get_euclidean_dist(line_to_classify, centroid)
            #dist = np.linalg.norm(line_to_classify - centroid)
            tree_dist_arr.append(euclidean_dist, tree)
        #sort from the smallest to bigest dist:
        trees_sorted_by_dist = sorted(tree_dist_arr, key=lambda tup: tup[0])
        topKtrees = trees_sorted_by_dist[:k]
        k_classification_array = []
        for tree in topKtrees:
            classification = getClassification(tree,line_to_classify,header)
            k_classification_array.append(classification)
        majority_classification = getMajorityClass(k_classification_array)
        if line_to_classify[0] == majority_classification:
            true += 1
        else:
            false += 1

        accuracy = true / len(test_data_without_header)
        return accuracy
        #chose the most classification
        #check if it is the correct one




p = 0.3 #is number of exmaples will be choosen from all the examples for each Tree
number_of_trees_N = 5 #number of trees
KNN(p, number_of_trees_N, 2) #todo: train on p from 0.3 to 0.7