from __future__ import division
import pandas
from TreeFunctions import *
from PrepareWalsData import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
# from ConvertToWalsTree import * 
from math import sin, cos, asin, sqrt
from LikelihoodFunction import *
# from Contact import * 

def reconstructStatesForNode(inputTree, node, states, matrix):
    tree = inputTree.copy()
    parent = findParent(inputTree, node)
    if parent == None:
        tree[node]['reconstructedStates'] = {}
        totalProbability = 0
        for state in states:
            if inputTree[node]['states'][state] == '?':
                tree[node]['reconstructedStates'] = UNASSIGNED
                return tree, 'Cannot do'
            totalProbability = totalProbability + inputTree[node]['states'][state]
        for state in states:
            tree[node]['reconstructedStates'][state] = inputTree[node]['states'][state]/totalProbability
        return tree, True
    else:
        if tree[parent]['reconstructedStates'] == UNASSIGNED:
            tree[node]['reconstructedStates'] = UNASSIGNED
            return tree, False
        else:
            tree[node]['reconstructedStates'] = {}
            branchLength = float(findBranchLength(node))
            for state in states:
                stateLikelihood = tree[node]['states'][state]
                if stateLikelihood == '?':
                    stateLikelihood = 1
                stateTotal = 0
                for parentState in states:
                    stateTotal = stateTotal + (tree[parent]['reconstructedStates'][parentState] * stateLikelihood * findTransitionProbability(parentState, state, states, matrix, branchLength)) 
                tree[node]['reconstructedStates'][state] = stateTotal
            total = 0
            for state in states:
                total = total + tree[node]['reconstructedStates'][state]
            for state in states:
                tree[node]['reconstructedStates'][state] = tree[node]['reconstructedStates'][state]/total				
        return tree, True

def reconstructStatesForAllNodes(inputTree, states, matrix):
    tree = inputTree.copy()
    done = False
    for node in tree:
        tree[node]['reconstructedStates'] = UNASSIGNED
    while done == False:
        done = True
        for node in tree:
            if tree[node]['reconstructedStates'] == UNASSIGNED:
                tree, nodeDone = reconstructStatesForNode(tree, node, states, matrix)
    # 				print node
    # 				print tree[node]
                if nodeDone == 'Cannot do':
                    return tree
                if not nodeDone:
                    done = False
    return tree

def produceListForNodeHeights(trees):
	nodeHeights = []
	for tree in trees:
		for node in tree.keys():
			height = findMaximumHeight(node)
			nodeHeights.append([node, height])
	nodeHeights = sorted(nodeHeights, key = lambda x: x[1], reverse = True)
	return nodeHeights

def produceDictionaryForNodeHeights(nodeHeightList):
	dictionary = {}
	for m in xrange(len(nodeHeightList)):
		try:
			x = dictionary[str(nodeHeightList[m][1])]
			if m < x[0]:
				x[0] = m
			if m > x[1]:
				x[1] = m
		except:
			dictionary[str(nodeHeightList[m][1])] = [m, m]
	return dictionary

# def produceListAndDictionaryForNodeHeights(trees):
# 	nodeHeightsList = produceListForNodeHeights(trees)
# 	nodeHeightsDictionary = produceDictionaryForNodeHeights(nodeHeightsList)
# 	return nodeHeightsList, nodeHeightsDictionary

# def prepareTrees(trees, numberUpTo = None, limitToIsos = True):
# 	if numberUpTo == None:
# 		numberUpTo = len(trees)
# 	for m in xrange(numberUpTo):
# 		tree = trees[m]
# 		tree = tree.strip('\n')
# 		tree = createTree(tree)
# 		if limitToIsos:
# 			tree = ensureAllTipsHaveIsoCodes(tree)
# 		trees[m] = tree
# 	return trees

# def reconstructStatesForAllTrees(trees, dataFrame, featureName, states, matrix, limitToIsos = True, numberUpTo = 'all'):
# 	result = {}
# 	if numberUpTo == 'all':
# 		numberUpTo = len(trees)
# 	for m in xrange(numberUpTo):
# 		tree = trees[m]
# # 		tree = tree.strip('\n')
# # 		tree = createTree(tree)
# # 		if limitToIsos:
# # 			tree = ensureAllTipsHaveIsoCodes(tree)
# 		tree = assignTipValuesByIso(tree, dataFrame, featureName)
# 		outputTree = calculateLikelihoodForAllNodes(tree, states, matrix)
# 		outputTree = reconstructStatesForAllNodes(outputTree, states, matrix)
# 		for node in outputTree:
# 			result[node] = outputTree[node]['reconstructedStates']
# 	return result

# def reconstructLocationsForAllTrees(trees, dataFrame, numberUpTo = 'all', limitToIsos = True):
# 	result = {}
# 	if numberUpTo == 'all':
# 		numberUpTo = len(trees)
# 	for m in xrange(numberUpTo):
# 		tree = trees[m]
# 		if limitToIsos:
# 			tree = ensureAllTipsHaveIsoCodes(tree)	
# 			tree = assignCoordinatesByIso(tree, dataFrame, 'latitude', 'longitude')	
# 		tree = reconstructLocationsForAllNodes(tree)
# 		for node in tree:
# 			result[node] = tree[node]
# 	return result
	