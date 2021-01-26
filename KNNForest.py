

#choose n*p(p is a parameter) for the training set:
import csv
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import sklearn

from ID3 import Node, fit, getAttributeCalumn, getMajorityClass, printTree
from CostSensitiveID3 import our_test_and_train


# #ex7 todo
# def get_normalized_centroid(df,random_data): #the centroid will have only the features in its vector
#     #pd.DataFrame(random_data).to_csv("C:/My Stuff/studies/2021a/AI/hw3/random_data.csv") #todo:remove
#     number_of_features = len(df[0])
#     len_of_random_data = len(random_data)
#     feature_average_array = []
#     for feature_place in range(1, number_of_features):  # for each feature #the first feature is diagnostic so i give it out
#         sum_for_feature = 0
#         for i in random_data:
#             feature_value_in_line_i = i[feature_place]
#             sum_for_feature += feature_value_in_line_i
#         average = sum_for_feature/len_of_random_data
#         feature_average_array.append(average)  # without the first feature which is diagnostic,is it okay?
#     return feature_average_array
#
# #todo: check



#todo : normalize the centroid so that the distance in the classification will be from all the vectore
#cheked
from ImprovedKNNForest import normalized_min_max_KNN


def get_centroid(df,random_data): #the centroid will have only the features in its vector
    #pd.DataFrame(random_data).to_csv("C:/My Stuff/studies/2021a/AI/hw3/random_data.csv") #todo:remove
    number_of_features = len(df[0])
    len_of_random_data = len(random_data)
    feature_average_array = []
    for feature_place in range(1, number_of_features):  # for each feature #the first feature is diagnostic so i let it out
        sum_for_feature = 0
        for i in random_data:
            feature_value_in_line_i = i[feature_place]
            sum_for_feature += feature_value_in_line_i
        average = sum_for_feature/len_of_random_data
        feature_average_array.append(average)  # without the first feature which is diagnostic,is it okay?
    return feature_average_array


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

#checked
def bulilt_N_trees(header,data_without_header,number_of_trees_N,p):
    tree_array = []
    n = len(data_without_header)
    for i in range(0, number_of_trees_N):
        number_of_random = round(p*n)
        random_data = random.sample(list(data_without_header), number_of_random)# no duplicate
        df = (header, random_data)
        centroid = get_centroid(df,random_data)#cheked
        node = Node()
        fit(df, node) #fit = algorithm ID3 thet used in ID3.py  #no pruning todo: maybe change to with pruning
        #printTree(node)
        tree_array.append((node, centroid)) #todo- try to print tree
    return tree_array


def get_dist_from_all_trees_centroid(trees_array,line_to_classify_without_classification):
    #printTree(trees_array[0][0])
    tree_dist_arr = []
    for tree, centroid in trees_array:
        #printTree(tree)
        euclidean_dist = get_euclidean_dist(line_to_classify_without_classification, centroid)
        # dist = np.linalg.norm(line_to_classify_without_classification - centroid)
        tree_dist_arr.append((euclidean_dist, tree))
    return tree_dist_arr

def getMajorityTreesClasification(header,line_to_classify_without_classification, topKtrees):
    k_classification_array = []
    for _, node in topKtrees:
        #printTree(node)
        classification = getClassification(header, node, line_to_classify_without_classification)
        k_classification_array.append(classification)
    majority_classification = getMajorityClass(k_classification_array)
    return majority_classification

# checked
def get_avg_array(data_without_header_and_without_first_col,len_of_data):
    array = np.array(data_without_header_and_without_first_col)
    sum_array = array.sum(axis=0)  # checkd
    avg_array = [x / len_of_data for x in sum_array]
    return avg_array

# cheked
def get_standard_deviation_array(avg_array,data_without_header_and_without_first_col,m):
    pow_inner_parentasis_list = []
    for line in data_without_header_and_without_first_col:
        a = np.matrix(line)
        b = np.matrix(avg_array)
        inner_parentasis = a - b  #(fi(xj)-mi) #for each line substract the avg
        inner_parentasis_list = inner_parentasis.tolist()[0]
        pow_inner_parentasis = [pow(x,2) for x in inner_parentasis_list]# (fi(xj)-mi)^2 #for each line substract the avg
        pow_inner_parentasis_list.append(pow_inner_parentasis)
    array = np.array(pow_inner_parentasis_list)
    sum_array = array.sum(axis=0)
    standard_deviation_array = [np.math.sqrt(x / (m - 1)) for x in sum_array]
    return standard_deviation_array



# cheked
def get_avg_and_standard_deviation_vec_from_data(data_without_header):
    pd.DataFrame(data_without_header).to_csv("C:/My Stuff/studies/2021a/AI/hw3/data_without_header.csv") #todo:remove
    #data = data_without_header
    #m = len(data_without_header)
    data_without_header_and_without_first_col =[] #without diagnosis
    for row in data_without_header:
        new_row_without_first_col = np.delete(row, [0])
        data_without_header_and_without_first_col.append(new_row_without_first_col)
    len_of_data = len(data_without_header)
    avg_array = get_avg_array(data_without_header_and_without_first_col,len_of_data)
    standard_deviation_array = get_standard_deviation_array(avg_array,data_without_header_and_without_first_col,len_of_data)

    return avg_array, standard_deviation_array

def KNN(data_without_header,test_data_without_header,header,p, number_of_trees_N, k):
    true = 0
    false = 0
    trees_array = bulilt_N_trees(header,data_without_header,number_of_trees_N,p)
    #printTree(trees_array[0][0]) #todo:remove
    #check the classification of examples in test.csv:
    #get_avg_and_standard_deviation_vec_from_data(data_without_header) #return vectores without the first diagnosis column
    for line_to_classify in test_data_without_header:
        real_classification_for_line = line_to_classify[0]
        line_to_classify_without_classification = np.delete(line_to_classify, [0]) #todo: test it removes the first element
        tree_dist_arr = get_dist_from_all_trees_centroid(trees_array, line_to_classify_without_classification)
        #printTree(tree_dist_arr[0][1])
        trees_sorted_by_dist = sorted(tree_dist_arr, key=lambda tup: tup[0])
        #printTree(trees_sorted_by_dist[0][1])
        topKtrees = trees_sorted_by_dist[:k]
        #printTree(topKtrees[0][1])
        majority_classification = getMajorityTreesClasification(header,line_to_classify_without_classification, topKtrees)

        if real_classification_for_line == majority_classification:
            true += 1
        else:
            false += 1

    accuracy = true / len(test_data_without_header)
    return accuracy
        #chose the most classification
        #check if it is the correct one



#cheked
def get_euclidean_dist(vec1,vec2):
    sum = 0
    for i,j in zip(vec1,vec2):
        sum += pow(i-j,2)
    return np.math.sqrt(sum)

# vec1 = [2,3]
# vec2 = [10,20]
# euclidean_dist = get_euclidean_dist(vec1,vec2)
# print(euclidean_dist)


# df = pd.read_csv("check_if_random.csv", header=0)
# data_without_header = df.to_numpy()
#
# random_data = random.sample(list(data_without_header), len(data_without_header))
#
# print("fin")




def k_fold_train_and_test_on_the_train_csv_forest(p, number_of_trees_in_comity_N, number_of_trees_to_classify_by_K,header,data_without_header,KNN_func):
    n_splits = 5
    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=311342422) #todo: check id is : 311342422
    kf.get_n_splits(data_without_header)
    accuracy_sum = 0
    for train_index, test_index in kf.split(data_without_header):
        train_data_without_header = []
        test_data_without_header = []
        for index in train_index:
            train_data_without_header.append(data_without_header[index])
        for index in test_index:
            test_data_without_header.append(data_without_header[index])
        accuracy = KNN_func(train_data_without_header,test_data_without_header,header,p,number_of_trees_in_comity_N,number_of_trees_to_classify_by_K)
        accuracy_sum += accuracy
    accuracy_mean = accuracy_sum/n_splits
    #print(accuracy_mean)
    return accuracy_mean


def expiriment_original_knn(header,data_without_header,KNN_func):
    ##graphs:

    array = [0.3, 0.4, 0.5, 0.6, 0.7]
    #array = [0.3]


    greatest_accuracy = 0
    number_of_trees_in_comity_N = 25 # number of trees# 20 is also okay
    for p in array:
        x = []
        y = []
        for number_of_trees_to_classify_by_K in range(15, number_of_trees_in_comity_N + 1):
            x.append(number_of_trees_to_classify_by_K)
            accuracy_mean = k_fold_train_and_test_on_the_train_csv_forest(p,number_of_trees_in_comity_N,number_of_trees_to_classify_by_K,header,data_without_header,KNN_func)
            #accuracy_mean = KNN(data_without_header,header,p, number_of_trees_in_comity_N, number_of_trees_to_classify_by_K)
            if accuracy_mean >= greatest_accuracy:
                print("p = ", p,"k = ", number_of_trees_to_classify_by_K, accuracy_mean)
                greatest_accuracy = accuracy_mean
            y.append(accuracy_mean)
        plt.plot(x, y)
        plt.xlabel('number_of_trees_to_classify_by(K)')
        plt.ylabel('accuracy')
        plt.title('p = ' + str(p) + str(KNN_func))
        plt.show()

def expiriment_original_knn_p_and_k(header,data_without_header,KNN_func,p,number_of_trees_to_classify_by_K):
    greatest_accuracy = 0
    number_of_trees_in_comity_N = number_of_trees_to_classify_by_K # number of trees# 20 is also okay
    x = []
    y = []
    x.append(number_of_trees_to_classify_by_K)
    accuracy_mean = k_fold_train_and_test_on_the_train_csv_forest(p,number_of_trees_in_comity_N,number_of_trees_to_classify_by_K,header,data_without_header,KNN_func)
    #accuracy_mean = KNN(data_without_header,header,p, number_of_trees_in_comity_N, number_of_trees_to_classify_by_K)
    if accuracy_mean >= greatest_accuracy:
        print("p = ", p,"k = ", number_of_trees_to_classify_by_K, accuracy_mean)
        greatest_accuracy = accuracy_mean
    y.append(accuracy_mean)
    plt.plot(x, y)
    plt.xlabel('number_of_trees_to_classify_by(K)')
    plt.ylabel('accuracy')
    plt.title('p = ' + str(p) + str(KNN_func))
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    df_test = pd.read_csv("test.csv", header=0)
    test_data_without_header = df_test.to_numpy()
    p = 0.4  # is number of exmaples will be choosen from all the examples for each Tree
    number_of_trees_in_comity_N = 18  # number of trees
    number_of_trees_to_classify_by_K = 17
    accuracy = KNN(data_without_header, test_data_without_header, header, p, number_of_trees_in_comity_N,
                   number_of_trees_to_classify_by_K)  # todo: train on p from 0.3 to 0.7
    print(accuracy)

#NOT RELEVANT:
# for i in range(0,20):
#     print("------------round",i,"-----------")
#     train,test,header = our_test_and_train()
#     extended_data = np.concatenate((train, test))
#     p= 0.4
#     k = 17
#     print("original KNN:", "p= ", p, ", k= ", k)
#     expiriment_original_knn_p_and_k(header,extended_data,KNN,p,k)



#checked with 1 1 1

# print("KNN on train and test")
# accuracy_sum = 0
# for i in range(0,20):
#     p = 0.4 #is number of exmaples will be choosen from all the examples for each Tree
#     number_of_trees_in_comity_N = 18 #number of trees
#     number_of_trees_to_classify_by_K = 17
#     accuracy = KNN(data_without_header,test_data_without_header,header,p, number_of_trees_in_comity_N, number_of_trees_to_classify_by_K) #todo: train on p from 0.3 to 0.7
#     print(accuracy)
#     accuracy_sum += accuracy





