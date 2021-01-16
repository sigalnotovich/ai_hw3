import numpy as np
import pandas as pd

eps = np.finfo(float).eps
from numpy import log2 as log



def get_entropy_before_division(df):
    entropy_node = 0  # Initialize Entropy
    values = df.diagnosis.unique()  # Unique objects - 'B', 'M'
    for value in values:
        fraction = df.diagnosis.value_counts()[value] / len(df.diagnosis)
        entropy_node += -fraction * np.log2(fraction)
        # print(entropy_node)  # todo: remove
    return entropy_node


#takes an attribute,looks at its values and devide into borders ,as we saw in the lecture,
#then add the sum over i to k of |Ei|/|E| * H(Ei) to the entropy_list
def find_entropy_for_different_divisions_for_attribute(df, attribute):
    #B_or_M = df.diagnosis.unique()  # This gives all 'B' and 'M'
    #print(B_or_M)
    # variables = df[attribute].unique()  # This gives all the values in the column of the attribute ,if there is a
    # duplicate, then only once
    # print(variables)
    entropy_list = []
    borders = dynamic_division(df, attribute)
    sum_of_lines = len(df)
    entropy_attribute = 0
    for border in borders:
        num_B = len(df[attribute][df[attribute] < border][df.diagnosis == 'B'])
        p_B = num_B/sum_of_lines
        num_M = len(df[attribute][df[attribute] < border][df.diagnosis == 'M'])
        p_M = num_M / sum_of_lines
        h_first_sun = -p_B * log(p_B + eps) - p_M * log(p_M + eps)
        num_B = len(df[attribute][df[attribute] > border][df.diagnosis == 'B'])
        p_B = num_B / sum_of_lines
        num_M = len(df[attribute][df[attribute] > border][df.diagnosis == 'M'])
        p_M = num_M / sum_of_lines
        h_second_sun = -p_B * log(p_B + eps) - p_M * log(p_M + eps)

        relation_of_chosen_from_all = len(df[attribute][df[attribute] < border])/len(df)
        new_entropy = relation_of_chosen_from_all * h_first_sun + relation_of_chosen_from_all * h_second_sun
        entropy_list.append((new_entropy, attribute, border))

    return entropy_list


def dynamic_division(df, attribute):
    variables = df[attribute].unique()
    # sort increaing order with lanbda
    variables = sorted(variables, key=lambda x: x, reverse=False)
    #print(variables)
    borders = lambda lst: [(lst[i] + lst[i + 1])/2 for i in range(0, len(lst) - 1)]
    #print(borders(variables))
    return borders(variables)

#get the entropy for all the attributes when each attribute devided to borders as we saw in the lecture
def get_division_options_entropy_list(df):
    division_options_entropy_list = []
    for attribute in df.keys()[1:]:
        # print(attribute)
        received_entropy_list_for_an_attribute = find_entropy_for_different_divisions_for_attribute(df, attribute)
        division_options_entropy_list += received_entropy_list_for_an_attribute
    return division_options_entropy_list



def MAX_IG(df):
    entropy_before_division = get_entropy_before_division(df)
    division_options_entropy_list = get_division_options_entropy_list(df)
    IG = [(entropy_before_division - entropy, attribute, border) for entropy, attribute, border in division_options_entropy_list]
    best_entropy_dif, best_attribute_name,  best_attribute_limit = max(IG,key=lambda item: item[0])  #for ex. (0.9457760422765744, 'fractal_dimension_mean', 0.050245)
    return best_entropy_dif, best_attribute_name,  best_attribute_limit  #for ex. (0.9457760422765744, 'fractal_dimension_mean', 0.050245)?


def get_subtable_under_limit(df, attribute, limit):
    return df[df[attribute] < limit].reset_index(drop=True)


def get_subtable_above_equal_limit(df, attribute, limit):
    return df[df[attribute] >= limit].reset_index(drop=True)



def buildTree(df, tree=None):
    Class = df.keys()[0]  # diagnosis

    # Here we build our decision tree

    # Get attribute with maximum information gain
    best_entropy_dif, best_attribute_name, best_attribute_limit = MAX_IG(df)

    print(best_attribute_name,best_attribute_limit)

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[best_attribute_name] = {}

    print("tree")
    print(tree)

    subtable_under_limit = get_subtable_under_limit(df, best_attribute_name, best_attribute_limit)
    subtable_under_limit.to_csv(r'C:\My Stuff\studies\2021a\AI\hw3\check\under_limit.csv')
    print("subtable_under_limit")
    print(subtable_under_limit)

    subtable_above_equal_limit = get_subtable_above_equal_limit(df, best_attribute_name, best_attribute_limit)
    subtable_above_equal_limit.to_csv(r'C:\My Stuff\studies\2021a\AI\hw3\check\above_equal_limit.csv')
    print("subtable_above_equal_limit")
    print(subtable_above_equal_limit)

    # clValue, counts = np.unique(subtable_under_limit, return_counts=True)
    # print("clValue" , clValue)
    # print("counts",counts)

    #clValue, counts = np.unique(subtable_above_equal_limit, return_counts=True)

##   if len(subtable_above_equal_limit['diagnosis'].unique().tolist()) == 1:
##       tree[best_attribute_name][best_attribute_limit] =
    # if len(counts) == 1:  # Checking purity of subset
    #     tree[best_attribute_name][best_attribute_limit] = clValue[0]
    # else:
    #     tree[best_attribute_name][best_attribute_limit] = buildTree(subtable_above_equal_limit)  # Calling the function recursively

    return tree

def getMajorityClass(df):
    list_of_B_and_M = df['diagnosis'].tolist()
    number_of_B = list_of_B_and_M.count('B')
    number_of_M = list_of_B_and_M.count('M')
    if number_of_B > number_of_M:
        return 'B'
    else:
        return 'M'


def ID3(df,node):

    majority_class = getMajorityClass(df)

    return TDIDT(df,majority_class,MAX_IG,node)


def TDIDT(df, majority_class ,MAX_IG,node):  #TDIDT(E, F, Default, SelectFeature)
    if df.empty == True:
        node.classification = majority_class
        return None

    majority_class = getMajorityClass(df)

    B_M_or_both = df['diagnosis'].unique().tolist()
    if len(B_M_or_both) == 1 : #only B or only M
        classification = B_M_or_both[0]
        node.classification = classification
        return None

    best_entropy_dif, best_attribute_name, best_attribute_limit = MAX_IG(df)
    node.partition_feature_and_limit = (best_attribute_name, best_attribute_limit)
    subtable_under_limit = get_subtable_under_limit(df, best_attribute_name, best_attribute_limit)
    node.left = Node()
    TDIDT(subtable_under_limit, majority_class ,MAX_IG,node.left)
    subtable_above_equal_limit = get_subtable_above_equal_limit(df, best_attribute_name, best_attribute_limit)
    node.right = Node()
    TDIDT(subtable_above_equal_limit, majority_class ,MAX_IG,node.right)

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.partition_feature_and_limit = None
        self.classification = None









df = pd.read_csv("train.csv", header=0)
node = Node()
ID3(df,node)
print(node)
#res = get_subtable(df,'perimeter_mean',54.09)
#print(res)

buildTree(df)
