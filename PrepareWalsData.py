import pandas as pd
import random
import numpy as np
from general import * 

random.seed(11)

# reads a file with a header, which has to include 'iso' (in lower case), and a feature name
def assignIsoValues(file):
	dataFrame = pd.read_csv(file, header = 0, sep =',', index_col = 'iso')	
	return dataFrame

def filterDataFrame(dataFrame, featureName, statesToInclude = []):
	filter = dataFrame[featureName].isin(statesToInclude)
	dataFrame = dataFrame[filter]
	return dataFrame

#  takes a dataframe with iso code as row name and feature name as column name. chooses among duplicates.
def lookUpValueForIso(iso, dataFrame, featureName):
	dataFrame = dataFrame[featureName]
	dataFrame = dataFrame.dropna()
	try:
		values = dataFrame.loc[iso]
	except:
		return "?"
	try:
		shape = values.shape
		return random.sample(values, 1)[0]
	except:
		return values
	return values


def findStates(dataFrame, featureName):
	states = dataFrame[featureName].dropna()
	return unique(states)
	