import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self):
        self.data = None

    def preprocess(self, path):
        self.data = pd.read_csv(path)
        # self.data = self.data.replace([np.inf, -np.inf], np.nan)
        #for i in col:
        #    if self.data[i].isnull().sum > 0:
        #        self.data[i].fillna(self.data[i].mode()[0], inplace=True)

        self.data.drop(['Unnamed: 0'], axis=1, inplace=True)
        columns = self.data.columns[self.data.columns != "Type"]
        # print("columns: ", columns)

        label = list(self.data["Type"])
        features = []
        for i in range(len(self.data)):
            feature = []
            for j in range(len(columns)):
                # print("this is ", i, "feature: ", feature)
                feature.append(self.data[columns[j]][i])

            features.append(feature)

        features = np.array(features, dtype=np.float64)

        label = np.array(label, dtype=int)

        return self.data, features, label