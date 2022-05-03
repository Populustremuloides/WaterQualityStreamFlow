import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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
df["random"] = np.random.normal(0,1,len(df[df.columns[0]]))
features = list(df.columns)[:16]
targets = list(df.columns[16:])


# for i in range(9):
#     newFeature = "randomized_" + str(df.columns[i])
#     features.append(newFeature)
#     oldData = list(df[df.columns[i]])
#     random.shuffle(oldData)
#     df[newFeature] = oldData

# add a "normal" dummy variable
newFeature = "normalRandom"
features.append(newFeature)
df[newFeature] = np.random.normal(0, 1, (len(df[df.columns[0]])))
print(df)

dataDict = {"repeat":[],"modeltype":[],"doc":[],"no3":[],"tn":[],"tp":[], "random":[]}

featureImportances = {"repeat":[],"modeltype":[],"target":[]}
for feature in features:
    featureImportances[feature] = []

X = df[features].to_numpy()
indices = list(range(X.shape[0]))
random.shuffle(indices)

trainIndices = np.asarray(indices[:630])
testIndices = np.asarray(indices[630:])
print("num overlapping indices")
print(len(list(set(trainIndices).intersection(set(testIndices)))))

numRepeats = 10

for repeat in range(0, numRepeats):
    print(repeat)
    dataDict["repeat"].append(repeat)
    dataDict["modeltype"].append("KNN")
    trainX = X[trainIndices]
    for target in targets:

        featureImportances["repeat"].append(repeat)
        featureImportances["modeltype"].append("KNN")
        featureImportances["target"].append(target)

        y = np.asarray(df[target])
        trainY = y[trainIndices]
        model = KNeighborsRegressor()
        model.fit(trainX, trainY)
        score = model.score(X[testIndices], y[testIndices])
        dataDict[target].append(score)

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("minmax_prediction_results_KNN.csv", index=False)
