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

def reconstructStatesForNode(inputTree, node, states, matrix, featureName):
    tree = inputTree.copy()
    parent = findParent(inputTree, node)
    if parent == None:
        tree[node][featureName]['reconstructedStates'] = {}
        totalProbability = 0
        for state in states:
            if inputTree[node][featureName]['states'][state] == '?':
                tree[node][featureName]['reconstructedStates'] = UNASSIGNED
                return tree, 'Cannot do'
            totalProbability = totalProbability + inputTree[node][featureName]['states'][state]
        for state in states:
            tree[node][featureName]['reconstructedStates'][state] = inputTree[node][featureName]['states'][state]/totalProbability
        return tree, True
    else:
        if tree[parent][featureName]['reconstructedStates'] == UNASSIGNED:
            tree[node][featureName]['reconstructedStates'] = UNASSIGNED
            return tree, False
        else:
            tree[node][featureName]['reconstructedStates'] = {}
            branchLength = float(findBranchLength(node))
            for state in states:
                stateLikelihood = tree[node][featureName]['states'][state]
                if stateLikelihood == '?':
                    stateLikelihood = 1
                stateTotal = 0
                for parentState in states:
                    stateTotal = stateTotal + (tree[parent][featureName]['reconstructedStates'][parentState] * stateLikelihood * findTransitionProbability(parentState, state, states, matrix, branchLength)) 
                tree[node][featureName]['reconstructedStates'][state] = stateTotal
            total = 0
            for state in states:
                total = total + tree[node][featureName]['reconstructedStates'][state]
            for state in states:
                tree[node][featureName]['reconstructedStates'][state] = tree[node][featureName]['reconstructedStates'][state]/total				
        return tree, True

def reconstructStatesForAllNodes(inputTree, states, matrix, featureName):
    tree = inputTree.copy()
    done = False
    for node in tree:
        tree[node][featureName]['reconstructedStates'] = UNASSIGNED
    while done == False:
        done = True
        for node in tree:
            if tree[node][featureName]['reconstructedStates'] == UNASSIGNED:
                tree, nodeDone = reconstructStatesForNode(tree, node, states, matrix, featureName)
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
	