#!/usr/bin/env python
# coding: utf-8

############Perceptron Algorithm ##########


import numpy as np
import pandas as pd
import itertools


# Perceptron Train function. The function accepts 4 params. 
# x -> dataset with all columns except the class label
# y -> class labels for each row in the input dataset
# max_iter -> no. of iterations to be performed. Here we are taking it as 20
# ld -> lambda value for regularisation. By default set to zero.
def perceptron_train(x, y, max_iter, ld=0):
    weight = np.zeros(x.shape[1])  # initialise a vector weight[d] to zero
    bias = 0  # initialise bias to zero

    for iter in range(max_iter):
        for element in range(len(x)):
            a = np.dot(weight, x[element]) + bias + ld * np.dot(weight, weight)  # activation score calculation
            if y[element] * a <= 0:  # checking if predicted value is different from the calculated one
                for i in range(len(weight)):
                    weight[i] += (y[element] * x[element][i]) - 2 * ld * weight[
                        i]  # during misclassification, update weight to w = w + y.x
                    bias += y[element]  # update bias to bias = bias+ y
    return (bias, weight)


# Used to convert the class labels to scalar values.
# This function is called in one-vs-rest approach to convert class corresponding to particular class to 1 
# and all others are set to -1
def convert_class_to_scalar(y, label):
    class_label = []
    for elem in y:
        if elem == label:
            class_label.append(1)
        else:
            class_label.append(-1)
    return (class_label)


# Used to convert the class labels to scalar values.
# This function is called in one-vs-one approach to convert class corresponding the subset of 2
# One is set to +1 and other to -1. All remaining class labels are ignored
def filter_out_class(y, labels):
    class_label = []
    for elem in y:
        if elem == labels[0]:
            class_label.append(1)
        elif elem == labels[1]:
            class_label.append(-1)
    return (class_label)


# function to filter out rows from the dataset which has class labels same as in the labels list.
def filter_out_rows(ds, labels):
    dim = ds.shape[1]
    result = np.empty((0, dim))
    for item in ds:
        if item[dim - 1] in labels:
            result = np.append(result, np.array([item]), axis=0)
    return (result)


# Perceptron test function for Test data predictions.
# Accepts 3 params:
# b -> bias from train dataset
# w -> weight from train dataset
# x -> Test dataset with all columns except the class label
# The function calculates activation score for each row against all classes
# whichever class has highest activation score, that class is assigned to that row of input 
def percetron_test(b, w, x):
    class_labels = ['class-1', 'class-2', 'class-3']
    activation_scores = []
    for element in range(len(x)):
        a = []
        for c in range((len(class_labels))):
            a.append(np.dot(w[c], x[element]) + b[c])  # activation score, a= w.x +b
        activation_scores.append("class-" + str(
            np.argmax(a) + 1))  # create the label list corresponding to the class with max activation score
    return activation_scores


# Function to calculate accuracy
# Accuracy = (no. of correctly classified entries)/total no. of entries in dataset
def calculate_accuracy(output, expected):
    true_count = 0
    for i in range(len(output)):
        if output[i] == expected[i]:
            true_count += 1;
    return (true_count / len(output))


# Read train and test data
dataset = pd.read_csv('train.data', header=None).values
test_set = pd.read_csv('test.data', header=None).values

activation_one_vs_rest = []

d = dataset.shape[1]
x = dataset[:, :d - 1]
y = dataset[:, d - 1]

test_dim = test_set.shape[1]
test_x = test_set[:, :test_dim - 1]
test_y = test_set[:, test_dim - 1]

# List of L2 regularisation params
l2_regularise = [0.01, 0.1, 1.0, 10.0, 100.0]
class_labels = ['class-1', 'class-2', 'class-3']

# One-vs-rest approach
bias_one_rest = []
weight_one_rest = []
for c in class_labels:
    y_scalar = convert_class_to_scalar(y, c)
    bias, weight = perceptron_train(x, y_scalar, 20)

    bias_one_rest.append(bias)
    weight_one_rest.append(weight.tolist())  # make a list of bias and weight for each class labels
    print("bias for one-vs-rest", bias)
    print("weight for one-vs-rest", weight)

# Test data check
# Calculating the accuracy over Test data
activation_one_vs_rest.append(percetron_test(bias_one_rest, weight_one_rest, test_x))
accuracy = calculate_accuracy(activation_one_vs_rest[0], test_y)
print("Accuracy for test data:", accuracy)

# L2 regularisation check
activation_test = []
activation_train = []

l2_bias_one_rest = []
l2_weight_one_rest = []

# Apply regularisation to test and train datasets
for l in l2_regularise:
    for c in class_labels:
        print(c)
        y_scalar = convert_class_to_scalar(y, c)
        bias, weight = perceptron_train(x, y_scalar, 20, l)
        l2_bias_one_rest.append(bias)
        l2_weight_one_rest.append(weight.tolist())
        print("bias after regularisation for one-vs-rest", bias)
        print("weight after regularisation for one-vs-rest", weight)

    activation_test.append(percetron_test(l2_bias_one_rest, l2_weight_one_rest, test_x))
    accuracy_test = calculate_accuracy(activation_test[0], test_y)
    print("\nfor regularisation param:", l, " - Accuracy of test set:", accuracy_test)

    activation_train.append(percetron_test(l2_bias_one_rest, l2_weight_one_rest, x))
    accuracy_train = calculate_accuracy(activation_train[0], y)
    print("\nfor regularisation param:", l, " - Accuracy of train set:", accuracy_train)

# One-to-one approach
# Subset of the three classes are formed and the bias and weights are checked for each combination
bias_one_one = []
weight_one_one = []
for subset in itertools.combinations(class_labels, 2):
    print("\n", subset)
    x_subset = filter_out_rows(dataset, subset)
    y_scalar = filter_out_class(x_subset[:, d - 1], subset)
    bias, weight = perceptron_train(x_subset[:, :d - 1], y_scalar, 20)
    bias_one_one.append(bias)
    weight_one_one.append(weight.tolist())
    print("bias for one-one", bias)
    print("weight for one-one", weight)
