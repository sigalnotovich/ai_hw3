import numpy as np
import pandas as pd
import csv
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ID3 import Node, fit


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



def getAttributeCalumn(header, partition_feature_and_limit):
    attribute_column = -1
    for i in range(0, len(header)):
        if header[i] == partition_feature_and_limit[0]:
            attribute_column = i
            break
    return attribute_column

def get_values_smaller_or_bigger_equal_to_partition(partition_feature_and_limit,prune_test_without_header,header):
    column = getAttributeCalumn(header,partition_feature_and_limit)
    smaller_then_attribute = []
    bigger_then_attribute = []
    for line in prune_test_without_header:
        if line[column] < partition_feature_and_limit[1]:
            smaller_then_attribute.append(line)
        else:
            bigger_then_attribute.append(line)
    return smaller_then_attribute,bigger_then_attribute


def prune(node,prune_test_without_header,header):
    if node.classification is not None:
        return node

    data_smaller_then_attribute, data_bigger_then_attribute = get_values_smaller_or_bigger_equal_to_partition(node.partition_feature_and_limit,prune_test_without_header,header)
    prune(node.left,data_smaller_then_attribute,header)
    prune(node.right,data_bigger_then_attribute,header)


    err_prune = 0
    err_no_prune = 0
    for line in prune_test_without_header:
        err_prune += evaluate(line[0],)
        err_no_prune += evaluate(line[0],getClassification(header,node,line))

    if err_prune < err_no_prune:
        node.classification =

    return


def costSensitiveID3():
    # 0.7 of the data to the tree
    #0.3 of the data to prune
    #then check loss on the test.csv
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    train_without_header, prune_test_without_header = train_test_split(data_without_header, test_size=0.3)

    df = (header, train_without_header)
    node = Node()
    fit(df, node)
    prune(node,prune_test_without_header,header)



costSensitiveID3()