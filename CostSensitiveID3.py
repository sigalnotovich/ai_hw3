import numpy as np
import pandas as pd
import csv
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from CostSensitiveID3anothertry import new_learn
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

    data_smaller_then_attribute, data_bigger_then_attribute = \
        get_values_smaller_or_bigger_equal_to_partition(node.partition_feature_and_limit,prune_test_without_header,header)
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


def k_fold_split_and_train_prune():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    n_splits = 5
    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=311342422) #todo: check id is : 311342422
    kf.get_n_splits(data_without_header)

    for train_index, test_index in kf.split(data_without_header):
        train_data = []
        prune_data = []
        for index in train_index:
            train_data.append(data_without_header[index])
        for index in test_index:
            prune_data.append(data_without_header[index])
    return  train_data,prune_data


def costSensitiveID3(header,data_without_header,test_df):
    train_without_header, prune_test_without_header = split_to_train_and_prune(data_without_header,0.5)
    df = (header, train_without_header)
    node = Node()
    fit(df, node)
    prune_node = prune(node,prune_test_without_header,header)
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
    test_data , train_data = train_test_split(all_lines, test_size=0.8)

    new_train = []
    new_test = []
    new_train = [y for x in [new_train, train_data] for y in x]  # new_train = new_train + train_data
    new_test = [y for x in [new_test, test_data] for y in x]  # new_test = new_test + test_data

    return new_train,new_test,header


def call_costSensitiveID3_with_original_data():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    df_test = pd.read_csv("test.csv", header=0)
    test_data_without_header = df_test.to_numpy()

    with open('test.csv', newline='') as t:
        reader = csv.reader(t)
        test_header = next(reader)

    test_df = (test_header, test_data_without_header)

    loss = costSensitiveID3(header,data_without_header,test_df)
    print(loss)



print("prune:")
call_costSensitiveID3_with_original_data()


