#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
from math import ceil
import sys

# preprocess
FOLD_SIZE = 5
REMOVE_OR_NOT = -1   # 1 is not remove, 2 is feature selection

want_list_index = []
want_list_name = []
discretization_index = []
class_name = []
class_index = -1
class_number = -1


def readFile(data):
    """
    reading file in csv file, initialize want_list_number
    """
    global discretization_index, want_list_index, class_index, class_number
    global REMOVE_OR_NOT
    with open(sys.argv[1], "r") as f:
        reader = csv.reader(f)
        for row in reader:
            row = map(float, row)
            data.append(row)

    with open(sys.argv[2], "r") as f:
        reader = csv.reader(f)
        want_list_index = map(int, reader.next())
        discretization_index = map(int, reader.next())
        class_index = int(reader.next()[0])
        class_number = int(reader.next()[0])

    REMOVE_OR_NOT = int(sys.argv[3])


def discretization(data, index):
    """
    10-bin discretization
    data and attribute index
    """
    # equal width
    Xmax = 0
    for i in xrange(len(data)):
        if data[i][index] == 999999:
            continue
        elif data[i][index] > Xmax:
            Xmax = data[i][index]

    # Xmax = data.max(axis=0)[index]
    Xmin = data.min(axis=0)[index]
    d = (Xmax - Xmin)*1.0 / 10
    for row in data:
        if d == 0:
            row[index] = 1
        else:
            row[index] = ceil((row[index] - Xmin) / d)
            if row[index] == 0:
                row[index] = 1


def fiveFold(foldLength, data, i):
    """
    five-fold data
    -->trainging data and testing data
    input: every fold length, shuffle data, fold index
    """
    start = i * foldLength
    end = (i + 1) * foldLength
    trainData = []
    testData = []
    for j in xrange(len(data)):
        if (j >= start and j < end) or (j >= end and i == FOLD_SIZE - 1):
            testData.append(data[j].tolist())
        else:
            trainData.append(data[j].tolist())
    return trainData, testData


def initWant(data):
    """
    initialize want_list_name and class_name
    """
    global want_list_name, class_name
    # initialize want_list_name
    for i in xrange(0, want_list_index[-1] + 1):
        want_list_name.append([])

    # count
    for i in xrange(len(data)):
        for j in want_list_index:
            # add into want_list_name to get attribute value of each attribute
            if data[i][j] not in want_list_name[j]:
                want_list_name[j].append(data[i][j])
            if data[i][class_index] not in class_name:
                class_name.append(data[i][class_index])


def calProb(trainData):
    """
    calculate probabilty dictionary
    """
    prob = {}
    probTotal = {}
    # count
    wantCount = 0
    for i in xrange(len(trainData)):
        if 999999 in trainData[i]:
            continue
        for j in want_list_index:
            tmpS = str(j) + "_" + str(trainData[i][j]) + "_" + str(trainData[i][class_index])
            if tmpS in prob:
                prob[tmpS] += 1
            else:
                prob[tmpS] = 1

        if trainData[i][class_index] in probTotal:
            probTotal[trainData[i][class_index]] += 1
        else:
            probTotal[trainData[i][class_index]] = 1
        wantCount += 1

    probN = dict(prob)
    probTotalN = dict(probTotal)

    # cal probability
    for i in want_list_index:
        for j in want_list_name[i]:
            for k in class_name:
                tmpS = str(i) + "_" + str(j) + "_" + str(k)
                if tmpS in prob:
                    prob[tmpS] = (prob[tmpS] + 1) * 1.0 / (probTotal[k] + len(want_list_name[i]))
                else:
                    prob[tmpS] = 1.0 / (probTotal[k] + len(want_list_name[i]))

    for k in class_name:
        probTotal[k] /= 1.0 * wantCount
    return prob, probTotal, probN, probTotalN


def testAccuracy(data, prob, probTotal, S):
    """
    test accuracy in SNB

        how to cal prob:
        for a_index in want_list_index:
            for a_index_value in want_list_name[a_index]:
                for cn in class_name:
                    tmpS = str(a_index) + "_" + str(a_index_value) + "_" + str(cn)
                    print prob[tmpS]
    """
    acc = 0
    for i in xrange(len(data)):
        pClass = []
        for cn in class_name:
            tmpP = probTotal[cn]
            for j in S:
                if data[i][j] == 999999:
                    continue
                tmpS = str(j) + "_" + str(data[i][j]) + "_" + str(cn)
                tmpP *= prob[tmpS]
            pClass.append(tmpP)
        if class_name[pClass.index(max(pClass))] == data[i][class_index]:
            acc += 1
    return acc * 1.0 / len(data)


def testDirchlet(data, probN, probTotalN, prob, probTotal, S, method):
    """
    test dirchlet is something kind of testAccuracy, but testing alpha
    argument
        method=1: direchlet
        method=2: general direchlet
    """
    alpha = [0] * len(S)
    for attrIndex in xrange(len(S)):
        maxAcc = 0
        maxAlpha = 0
        alpha[attrIndex] = -1   # now look at this attribute
        for a in xrange(1, 51):
            acc = 0
            for i in xrange(len(data)):
                pClass = []
                for cn in class_name:
                    tmpP = probTotal[cn]
                    for j in S:
                        if data[i][j] == 999999:
                            continue
                        tmpS = str(j) + "_" + str(data[i][j]) + "_" + str(cn)
                        if alpha[S.index(j)] == 0 and method == 2:
                            tmpP *= prob[tmpS]
                        else:
                            if tmpS in probN:
                                tmpP *= (probN[tmpS] + a) * 1.0 / (probTotalN[cn] + a * (len(want_list_name[j])))
                            else:
                                tmpP *= a * 1.0 / (probTotalN[cn] + a * (len(want_list_name[j])))
                    pClass.append(tmpP)
                if class_name[pClass.index(max(pClass))] == data[i][class_index]:
                    acc += 1
            if acc > maxAcc:
                maxAcc = acc
                maxAlpha = a
        alpha[attrIndex] = maxAlpha
        if method == 1:
            alpha = [alpha[attrIndex]] * len(S)
            print "direchlet alpha: ", alpha
            break

        if method == 2:
            print "general direchlet alpha: ", alpha

    # cal probabilty again
    probNew = dict(probN)
    for i in S:
        for j in want_list_name[i]:
            for k in class_name:
                tmpS = str(i) + "_" + str(j) + "_" + str(k)
                if tmpS in probN:
                    probNew[tmpS] = (probNew[tmpS] + alpha[S.index(i)]) * 1.0 / (probTotalN[k] + alpha[S.index(i)] * (len(want_list_name[i])))
                else:
                    probNew[tmpS] = alpha[S.index(i)] * 1.0 / (probTotalN[k] + alpha[S.index(i)] * (len(want_list_name[i])))

    return probNew


def SNB(index, trainData, prob, probTotal):
    """
    SNB: 1 for no delete attribute, 2 for feature selection
    """
    S = []
    Y = list(want_list_index)
    totalMaxAcc = 0
    while Y:
        maxAcc = 0
        maxName = -1
        for i in Y:
            S.append(i)
            print S
            print "--------------"
            tmpAcc = testAccuracy(trainData, prob, probTotal, S)
            print tmpAcc
            print "--------------"

            if tmpAcc > maxAcc:
                maxAcc = tmpAcc
                maxName = i
            S.remove(i)
            print "maxAcc: " + str(maxAcc) + " maxName: " + str(maxName)

        # check if need to do the fature selection
        if REMOVE_OR_NOT == 1:
            S.append(maxName)
            Y.remove(maxName)
        else:
            if maxAcc > totalMaxAcc:
                S.append(maxName)
                Y.remove(maxName)
                totalMaxAcc = maxAcc
            else:
                break

        print "========================================="
        print "========================================="
    return S


def main():
    if len(sys.argv) != 5:
        print "[Usage]: python *.py data data_index method random_seed"
        print "method 1: No feature selection"
        print "method 2: feature selection"
        return

    data = []
    readFile(data)
    data = np.array(data)   # change to numpy array

    # discretization
    for i in discretization_index:
        discretization(data, i)

    # shuffle the list
    np.random.seed(int(sys.argv[4]))
    np.random.shuffle(data)

    # five-fold cross validation + SNB
    initWant(data)
    foldLength = (int)(len(data) / FOLD_SIZE)
    testDataAcc = [0] * 3
    for i in xrange(FOLD_SIZE):
        trainData, testData = fiveFold(foldLength, data, i)

        # calculate probability
        # probability[attribute_atrributeValue_classValue]
        # probabilty_Total[classValue]
        prob, probTotal, probN, probTotalN = calProb(trainData)
        want_list = SNB(REMOVE_OR_NOT, trainData, prob, probTotal)

        # only laplace estimate
        testDataAcc[0] += testAccuracy(testData, prob, probTotal, want_list)

        # Direchlet, method argument == 1
        probNew = testDirchlet(trainData, probN, probTotalN, prob, probTotal, want_list, 1)
        testDataAcc[1] += testAccuracy(testData, probNew, probTotal, want_list)

        # general Direchlet, method argument == 2
        probNew = testDirchlet(trainData, probN, probTotalN, prob, probTotal, want_list, 2)
        testDataAcc[2] += testAccuracy(testData, probNew, probTotal, want_list)

    print "Testing data Accuracy: "
    print "Laplace estimate: " + str(testDataAcc[0] / FOLD_SIZE)
    print "Direchlet estimate: " + str(testDataAcc[1] / FOLD_SIZE)
    print "General Direchlet estimate: " + str(testDataAcc[2] / FOLD_SIZE)

    return 0

if __name__ == '__main__':
    main()
