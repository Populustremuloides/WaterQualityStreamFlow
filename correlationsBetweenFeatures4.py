import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def sqrtData(data):
    values = np.array(data)
    values = values + 0.0000000001
    values = np.sqrt(values)
    return values

df = pd.read_csv("epa_minmax_complete_TRAIN.csv")
df = df.drop("sort", axis=1)
# df = df.dropna()
for i, col in enumerate(df.columns):
    print(i, col)
    print(np.sum(~pd.isna(df[col])))
# quit()

df = df.dropna()
no3 = np.asarray(df["no3"])
no3 = np.exp(no3) - 1
no3 = sqrtData(sqrtData(no3))
df["no3"] = no3
for i in range(1,16):
    # newFeature = str(df.columns[i])
    oldData = list(df[df.columns[i]])
    random.shuffle(oldData)
    df[df.columns[i]] = oldData
df["random"] = np.random.normal(0,1,len(df[df.columns[0]]))


features = list(df.columns)[:16]
targets = list(df.columns[16:])

newFeature = "normalRandom"
features.append(newFeature)
df[newFeature] = np.random.normal(0, 1, (len(df[df.columns[0]])))

dataDict = {"repeat":[],"modeltype":[],"doc":[],"no3":[],"tn":[],"tp":[],"random":[]}

X = df[features].to_numpy()

indices = list(range(X.shape[0]))
random.shuffle(indices)

featureImportances = {"repeat":[],"modeltype":[],"target":[]}
for feature in features:
    featureImportances[feature] = []

trainIndices = np.asarray(indices[:630])
testIndices = np.asarray(indices[630:])
print("num overlapping indices")
print(len(list(set(trainIndices).intersection(set(testIndices)))))

numRepeats = 100

for repeat in range(0,numRepeats):
    print(repeat)
    dataDict["repeat"].append(repeat)
    dataDict["modeltype"].append("RFR")
    trainX = X[trainIndices]
    for target in targets:

        featureImportances["repeat"].append(repeat)
        featureImportances["modeltype"].append("RFR")
        featureImportances["target"].append(target)

        y = np.asarray(df[target])
        trainY = y[trainIndices]
        model = RandomForestRegressor(n_estimators=100)
        model.fit(trainX, trainY)
        score = model.score(X[testIndices], y[testIndices])
        dataDict[target].append(score)

        importances = model.feature_importances_
        for j, feature in enumerate(features):
            featureImportances[feature].append(importances[j])

for repeat in range(numRepeats, numRepeats*2):
    print(repeat)
    dataDict["repeat"].append(repeat)
    dataDict["modeltype"].append("GBR")
    trainX = X[trainIndices]
    for target in targets:

        featureImportances["repeat"].append(repeat)
        featureImportances["modeltype"].append("GBR")
        featureImportances["target"].append(target)

        y = np.asarray(df[target])
        trainY = y[trainIndices]
        if target == "tn":
            model = GradientBoostingRegressor(n_estimators=50)
        else:
            model = GradientBoostingRegressor(n_estimators=100)
        model.fit(trainX, trainY)
        score = model.score(X[testIndices], y[testIndices])
        dataDict[target].append(score)

        importances = model.feature_importances_
        for j, feature in enumerate(features):
            featureImportances[feature].append(importances[j])

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("minmax_prediction_results_randomX.csv", index=False)

df = pd.DataFrame.from_dict(featureImportances)
df.to_csv("minmax_prediction_featureImportances_randomX.csv", index=False)
