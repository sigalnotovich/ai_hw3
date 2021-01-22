import numpy as np
import pandas as pd
import csv
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ID3 import Node, fit, loss_func, getAttributeCalumn, printTree


#checked in ID3
def getClassification(header,node,line_to_classify):
    #printTree(node)
    if node.classification is not None:
        return node.classification
    else:
        # there is partition here
        attribute_column = getAttributeCalumn(header, node)
        value = line_to_classify[attribute_column-1]
        if value < node.partition_feature_and_limit[1]:#in new line i omited the first line with the classafication, thaths why -1
            return getClassification(header, node.left, line_to_classify)  # under limit
        else:
            return getClassification(header, node.right,line_to_classify)  # above or equal to limit


# cheked
def getAttributefeatureCalumn(header, partition_feature_and_limit):
    attribute_column = -1
    for i in range(0, len(header)):
        if header[i] == partition_feature_and_limit[0]:
            attribute_column = i
            break
    return attribute_column

def get_values_smaller_or_bigger_equal_to_partition(partition_feature_and_limit,prune_test_without_header,header):
    column = getAttributefeatureCalumn(header,partition_feature_and_limit)
    smaller_then_attribute = []
    bigger_then_attribute = []
    for line in prune_test_without_header:
        val = line[column]
        if line[column] < partition_feature_and_limit[1]:
            smaller_then_attribute.append(line)
        else:
            bigger_then_attribute.append(line)
    return smaller_then_attribute,bigger_then_attribute

def evaluate(real_class, tree_classification):
    if real_class == tree_classification:
        return 0
    if real_class == 'M' and tree_classification == 'B': #B is helathy
        return 9
    if real_class == 'B' and tree_classification == 'M':
        return 1


def prune(node,prune_test_without_header,header):
    if node.classification is not None:
        return node

    data_smaller_then_attribute, data_bigger_then_attribute = get_values_smaller_or_bigger_equal_to_partition(node.partition_feature_and_limit,prune_test_without_header,header)
    node.left = prune(node.left,data_smaller_then_attribute,header)
    node.right = prune(node.right,data_bigger_then_attribute,header)


    err_prune = 0
    err_no_prune = 0
    for line in prune_test_without_header:
        err_prune += evaluate(line[0],node.majority)
        err_no_prune += evaluate(line[0],getClassification(header,node,line))

    if err_prune < err_no_prune:
        node.classification = node.majority

    return node

def split_to_train_and_prune(data_without_header, train_present):
    n = int(len(data_without_header) * train_present)
    train_data = data_without_header[0:n]
    prune_data = data_without_header[n:]
    return train_data, prune_data


def costSensitiveID3(header,data_without_header):
    # 0.7 of the data to the tree
    #0.3 of the data to prune
    #then check loss on the test.csv


    train_without_header, prune_test_without_header = split_to_train_and_prune(data_without_header,0.5)
    #train_without_header, prune_test_without_header = train_test_split(data_without_header, test_size=0.3)

    df = (header, train_without_header)
    node = Node()
    fit(df, node)
    print("tree:")
    printTree(node)
    prune_node = prune(node,prune_test_without_header,header)
    print("pruned tree:")
    printTree(prune_node)
    df_test = pd.read_csv("test.csv", header=0)
    test_data_without_header = df_test.to_numpy()

    with open('test.csv', newline='') as t:
        reader = csv.reader(t)
        test_header = next(reader)

    test_df = (test_header, test_data_without_header)

    loss = loss_func(test_df, prune_node)
    return loss



def new_test_and_train():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    df_test = pd.read_csv("test.csv", header=0)
    test_data_without_header = df_test.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)


    all_lines = [y for x in [data_without_header, test_data_without_header] for y in x] # all_lines = data_without_header + test_data_without_header
    test_data , train_data = train_test_split(all_lines, test_size=0.7)

    new_train = []
    new_test = []
    new_train.append(header)
    new_train = [y for x in [new_train, train_data] for y in x]  # new_train = new_train + train_data
    new_test.append(header)
    new_test = [y for x in [new_test, test_data] for y in x]  # new_test = new_test + test_data

    return new_train,new_test

new_train,new_test = new_test_and_train()

def call_costSensitiveID3_with_original_data():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    loss = costSensitiveID3(header,data_without_header)
    print(loss)


call_costSensitiveID3_with_original_data()