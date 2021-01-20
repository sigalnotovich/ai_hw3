

#choose n*p(p is a parameter) for the training set:
import csv
import pandas as pd
import random

from ID3 import Node, fit

#todo : normalize the centroid so that the distance in the classification will be from all the vectore
def get_centroid(df,random_data):
    pd.DataFrame(random_data).to_csv("C:/My Stuff/studies/2021a/AI/hw3/random_data.csv")
    number_of_features = len(df[0])
    len_of_random_data = len(random_data)
    feature_average_array = []
    feature_average_array.append(' ')  # for the first feature of the diagnosis - so it will be organized
    for feature_place in range(1, number_of_features):  # for each feature #the first feature is diagnostic so i give it out
        sum_for_feature = 0
        for i in random_data:
            feature_value_in_line_i = i[feature_place]
            sum_for_feature += feature_value_in_line_i
        average = sum_for_feature/len_of_random_data
        feature_average_array.append(average)  # without the first feature which is diagnostic,is it okay?
    return feature_average_array



def choose_number_of_examples_for_training(p,number_of_trees_N):
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
        fit(df, node) #fit = algorithm ID3 thet used in ID3.py
        tree_array.append(node, centroid)
    # now that i have all the trees :
    # choose the k nearest trees to the centorid :
    # for each tree check the right answer
    # then take the answer the most give
    # check if it is the right answer - then do it for all the examples in the test.csv to check the accuracy



p = 0.3 #is number of exmaples will be choosen from all the examples for each Tree
number_of_trees_N = 5 #number of trees
choose_number_of_examples_for_training(p,number_of_trees_N) #todo: train on p from 0.3 to 0.7