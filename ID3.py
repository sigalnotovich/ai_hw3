import numpy as np
import pandas as pd
import csv
import sklearn.model_selection

eps = np.finfo(float).eps
from numpy import log2 as log


def get_entropy_before_division(df, list_of_B_and_M):
    entropy_node = 0
    # for B:
    sum_of_values = len(list_of_B_and_M)
    sum_of_B = len([x for x in list_of_B_and_M if x == 'B'])
    sum_of_M = len([x for x in list_of_B_and_M if x == 'M'])
    fraction_B = sum_of_B / sum_of_values
    fraction_M = sum_of_M / sum_of_values
    entropy_B = -fraction_B * np.log2(fraction_B)
    entropy_M = -fraction_M * np.log2(fraction_M)
    entropy = entropy_B + entropy_M
    return entropy


# takes an attribute,looks at its values and devide into borders ,as we saw in the lecture,
# then add the sum over i to k of |Ei|/|E| * H(Ei) to the entropy_list
# checked
def find_entropy_for_different_divisions_for_attribute(df, attribute, unsorted_list_of_values, len_of_table,
                                                       list_of_B_and_M):
    entropy_list = []

    # get_borders on sorted values of the attribute: checked
    borders_func = lambda list: [(list[i] + list[i + 1]) / 2 for i in
                                 range(0, len(list) - 1)]
    list_of_sorted_values = sorted(unsorted_list_of_values, key=lambda x: x, reverse=False)
    list_of_sorted_values_no_duplicates = list(dict.fromkeys(list_of_sorted_values))
    borders = borders_func(list_of_sorted_values_no_duplicates)
    # print(list_of_values)
    # print(borders)

    # use unsorted values so that list_of_B_and_M will match the values of the attribute
    unsorted_values_with_B = []
    unsorted_values_with_M = []
    # get all the values with B: checked
    for i in range(0, len(unsorted_list_of_values)):
        if list_of_B_and_M[i] == 'B':
            unsorted_values_with_B.append(unsorted_list_of_values[i])
        else:
            unsorted_values_with_M.append(unsorted_list_of_values[i])
    # print(unsorted_values_with_B)
    # print(unsorted_values_with_M)

    sorted_values_with_B = sorted(unsorted_values_with_B, key=lambda x: x, reverse=False)
    sorted_values_with_M = sorted(unsorted_values_with_M, key=lambda x: x, reverse=False)

    # print(sorted_values_with_B)
    # print(sorted_values_with_M)

    number_of_rows = len(df[1])
    # borders = dynamic_division(df, attribute,list_of_values)
    entropy_attribute = 0
    for border in borders:
        num_B_smaller_then_border = len([value for value in sorted_values_with_B if value < border])
        num_M_smaller_then_border = len([value for value in sorted_values_with_M if value < border])
        number_of_rows_smaller_then_border = num_B_smaller_then_border + num_M_smaller_then_border
        p_B_smaller_then_border = num_B_smaller_then_border / number_of_rows_smaller_then_border
        p_M_smaller_then_border = num_M_smaller_then_border / number_of_rows_smaller_then_border
        entropy_first_sun = -p_B_smaller_then_border * np.log2(p_B_smaller_then_border + eps) \
                      - p_M_smaller_then_border * np.log2(p_M_smaller_then_border + eps)

        num_B_bigger_equal_then_border = len([value for value in sorted_values_with_B if value >= border])
        num_M_bigger_equal_then_border = len([value for value in sorted_values_with_M if value >= border])
        number_of_rows_bigger_equal_then_border = num_B_bigger_equal_then_border + num_M_bigger_equal_then_border
        p_B_bigger_equal_then_border = num_B_bigger_equal_then_border / number_of_rows_bigger_equal_then_border
        p_M_bigger_equal_then_border = num_M_bigger_equal_then_border / number_of_rows_bigger_equal_then_border
        entropy_second_sun = -p_B_bigger_equal_then_border * np.log2(p_B_bigger_equal_then_border + eps) \
                       - p_M_bigger_equal_then_border * np.log2(p_M_bigger_equal_then_border + eps)

        sum_of_smaller_then_border = num_B_smaller_then_border + num_M_smaller_then_border
        add_to_entropy_first_sun = sum_of_smaller_then_border / number_of_rows * entropy_first_sun

        sum_of_bigger_equal_then_border = num_B_bigger_equal_then_border + num_M_bigger_equal_then_border
        add_to_entropy_second_sun = sum_of_bigger_equal_then_border / number_of_rows * entropy_second_sun

        new_entropy = add_to_entropy_first_sun + add_to_entropy_second_sun
        entropy_list.append((new_entropy, attribute, border))

    return entropy_list



# get the entropy for all the attributes when each attribute devided to borders as we saw in the lecture
def get_best_IG(df, entropy_before_division):
    division_options_entropy_list = []
    number_of_columns_in_file = len(df[1][0])
    list_of_B_and_M = [line[0] for line in df[1]]  # distinct, in order
    best_IG_dif = 0
    best_attribute_name = ""
    best_attribute_limit = 0

    # go over all the attributes in file - checked
    for i in range(1, number_of_columns_in_file):
        attribute = df[0][i]  # checked
        # print(attribute)
        list_of_values_of_attribute_i = [x[i] for x in df[1]]  # chcked
        # print(list_of_values_of_attribute_i)
        received_entropy_list_for_an_attribute = find_entropy_for_different_divisions_for_attribute(df, attribute,
                                                                                                    list_of_values_of_attribute_i,
                                                                                                    number_of_columns_in_file,
                                                                                                    list_of_B_and_M)

        IG_list_for_attribute = [(entropy_before_division - entropy, attribute, border) for entropy, attribute, border in received_entropy_list_for_an_attribute]
        IG_dif, attribute_name, attribute_limit = max(IG_list_for_attribute, key=lambda item: item[0])
        if IG_dif >= best_IG_dif:
            best_IG_dif, best_attribute_name, best_attribute_limit = IG_dif, attribute_name, attribute_limit

    return best_IG_dif, best_attribute_name, best_attribute_limit

def MAX_IG(df,list_of_B_and_M):
    entropy_before_division = get_entropy_before_division(df,list_of_B_and_M)

    best_IG_dif, best_attribute_name, best_attribute_limit = get_best_IG(df, entropy_before_division)
    return best_IG_dif, best_attribute_name, best_attribute_limit #for ex. (0.9457760422765744, 'fractal_dimension_mean', 0.050245)?



def get_subtable_under_and_above_equal_to_limit(df, attribute, limit):
    subtable_under_limit = []
    subtable_above_equal_to_limit = []
    attribute_column = -1
    for i in range(0, len(df[0])):
        if df[0][i] == attribute:
            attribute_column = i
            break

    for j in df[1]:
        if j[attribute_column] < limit:
            subtable_under_limit.append(j)
        else:
            subtable_above_equal_to_limit.append(j)

    df_under_limit = (df[0], subtable_under_limit)
    df_above_equal_to_limit = (df[0], subtable_above_equal_to_limit)
    # print(df_under_limit[1][3])
    # print(df_above_equal_to_limit[1][3])
    return df_under_limit, df_above_equal_to_limit


# def get_subtable_under_limit(df, attribute, limit):
#     return df[df[attribute] < limit].reset_index(drop=True)
#
#
# def get_subtable_above_equal_limit(df, attribute, limit):
#     return df[df[attribute] >= limit].reset_index(drop=True)


# def buildTree(df, tree=None):
#     Class = df.keys()[0]  # diagnosis
#
#     # Here we build our decision tree
#
#     # Get attribute with maximum information gain
#     best_entropy_dif, best_attribute_name, best_attribute_limit = MAX_IG(df)
#
#     print(best_attribute_name,best_attribute_limit)
#
#     # Create an empty dictionary to create tree
#     if tree is None:
#         tree = {}
#         tree[best_attribute_name] = {}
#
#     print("tree")
#     print(tree)
#
#
#     subtable_under_limit = get_subtable_under_limit(df, best_attribute_name, best_attribute_limit)
#     subtable_under_limit.to_csv(r'C:\My Stuff\studies\2021a\AI\hw3\check\under_limit.csv')
#     #print("subtable_under_limit")
#     #print(subtable_under_limit)
#
#     subtable_above_equal_limit = get_subtable_above_equal_limit(df, best_attribute_name, best_attribute_limit)
#     subtable_above_equal_limit.to_csv(r'C:\My Stuff\studies\2021a\AI\hw3\check\above_equal_limit.csv')
#     #print("subtable_above_equal_limit")
#     #print(subtable_above_equal_limit)
#
#     # clValue, counts = np.unique(subtable_under_limit, return_counts=True)
#     # print("clValue" , clValue)
#     # print("counts",counts)
#
#     #clValue, counts = np.unique(subtable_above_equal_limit, return_counts=True)
#
# ##   if len(subtable_above_equal_limit['diagnosis'].unique().tolist()) == 1:
# ##       tree[best_attribute_name][best_attribute_limit] =
#     # if len(counts) == 1:  # Checking purity of subset
#     #     tree[best_attribute_name][best_attribute_limit] = clValue[0]
#     # else:
#     #     tree[best_attribute_name][best_attribute_limit] = buildTree(subtable_above_equal_limit)  # Calling the function recursively
#
#     return tree

# checked
def getMajorityClass(list_of_B_and_M):
    number_of_B = list_of_B_and_M.count('B')
    number_of_M = list_of_B_and_M.count('M')
    if number_of_B > number_of_M:
        return 'B'
    else:
        return 'M'


# checked
def ID3(df, node,early_pruning_parameter):
    list_of_B_and_M = [line[0] for line in df[1]]  # todo:maybe pass it to the function - used a lot
    # print(list_of_B_and_M)
    # pd.DataFrame(list_of_B_and_M).to_csv("C:/My Stuff/studies/2021a/AI/hw3/list_of_B_and_M.csv")

    majority_class = getMajorityClass(list_of_B_and_M)

    return TDIDT(df, majority_class, MAX_IG, node,early_pruning_parameter)


# df[0] = heaser of the file, df[1] = all the other data in the file
def TDIDT(df, majority_class_df_under_limit, MAX_IG, node,early_pruning_parameter):  # TDIDT(E, F, Default, SelectFeature)
    if len(df[1]) == 0:
        node.classification = majority_class_df_under_limit
        return None

    list_of_B_and_M = [line[0] for line in df[1]]

    B_M_or_both = set(list_of_B_and_M)
    if len(B_M_or_both) == 1:  # only B or only M
        classification = B_M_or_both.pop()
        node.classification = classification
        return None

    if early_pruning_parameter is not None:
        if len(df[1]) <= early_pruning_parameter:
            node.classification = majority_class_df_under_limit
            return None
    # majority_class = getMajorityClass(list_of_B_and_M) #todo: do i need it here?

    # choose best festure and best limit:
    best_entropy_dif, best_attribute_name, best_attribute_limit = MAX_IG(df, list_of_B_and_M)
    # save at node the best attribute and limit for it
    node.partition_feature_and_limit = (best_attribute_name, best_attribute_limit)

    df_under_limit, df_above_equal_to_limit = get_subtable_under_and_above_equal_to_limit(df, best_attribute_name,
                                                                                          best_attribute_limit)
    # pd.DataFrame(df_under_limit[1]).to_csv("C:/My Stuff/studies/2021a/AI/hw3/df_under_limit.csv")
    # pd.DataFrame(df_above_equal_to_limit[1]).to_csv("C:/My Stuff/studies/2021a/AI/hw3/df_above_equal_to_limit.csv")

    # construct left and right node
    node.left = Node()
    list_of_B_and_M_under_limit = [line[0] for line in df_under_limit[1]]
    majority_class_df_under_limit = getMajorityClass(list_of_B_and_M_under_limit)
    TDIDT(df_under_limit, majority_class_df_under_limit, MAX_IG, node.left,early_pruning_parameter)

    node.right = Node()
    list_of_B_and_M_above_equal_to_limit = [line[0] for line in df_above_equal_to_limit[1]]
    majority_above_equal_to_limit = getMajorityClass(list_of_B_and_M_above_equal_to_limit)
    TDIDT(df_above_equal_to_limit, majority_above_equal_to_limit, MAX_IG, node.right,early_pruning_parameter)

    return None


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.partition_feature_and_limit = None
        self.classification = None

def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        if node.partition_feature_and_limit is not None:
            print(' ' * 4 * level + '->', node.partition_feature_and_limit)
        if node.classification is not None:
            print(' ' * 4 * level + '->', node.classification)
        printTree(node.right, level + 1)


# header = np.genfromtxt('train.csv', dtype=float, delimiter=',', names=True)

def fit(df,early_pruning_parameter = None):
    #print(df[0])
    #print(df[1])
    node = Node()
    ID3(df, node,early_pruning_parameter)
    return node
    # print(node)


# checked
def getAttributeCalumn(header, node):
    attribute_column = -1
    for i in range(0, len(header)):
        if header[i] == node.partition_feature_and_limit[0]:
            attribute_column = i
            break
    return attribute_column


def predict(node):
    true = 0
    false = 0
    df = pd.read_csv("test.csv", header=0)
    data_without_header = df.to_numpy()

    with open('test.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    for data_line in data_without_header:
        if return_prediction_good_or_bad_for_a_line_of_data(header, node, data_line):
            true += 1
        else:
            false += 1

    accuracy = true / len(df)
    return accuracy


def return_prediction_good_or_bad_for_a_line_of_data(header, node, data_line):
    if node.classification is not None:
        if data_line[0] == node.classification:
            return True
        else:
            return False
    else:
        # there is partition here
        attribute_column = getAttributeCalumn(header, node)
        if data_line[attribute_column] < node.partition_feature_and_limit[1]:
            return return_prediction_good_or_bad_for_a_line_of_data(header, node.left, data_line)  # under limit
        else:
            return return_prediction_good_or_bad_for_a_line_of_data(header, node.right,
                                                                    data_line)  # above or equal to limit


def ex1():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    df = (header, data_without_header)
    node = fit(df)
    accuracy = predict(node)
    #printTree(node)
    print(accuracy)


def ex3():
    df = pd.read_csv("train.csv", header=0)
    data_without_header = df.to_numpy()

    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    # df = (header, data_without_header)
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle= True, random_state= 123456789)
    kf.get_n_splits(data_without_header)
    train_data = []
    test_data = []
    for train_index, test_index in kf.split(data_without_header):
        for index in train_index:
            train_data.append(data_without_header[index])
        for index in test_index:
            test_data.append(data_without_header[index])
        df = (header, train_data)
        node = fit(df)
        accuracy = predict(node)
        # printTree(node)
        print(accuracy)





    #print("fin")



# for i in range(1,40):
#     node = fit(i)
#     print(i)
#     accuracy = predict(node)
#     #printTree(node)
#     print(accuracy)
#ex1 - todo: remove Comment
ex1()
#ex3()

