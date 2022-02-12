import numpy as np
import pandas as pd
import pickle
from scipy.io import arff
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV

class DecisionTree:

    model_type = "decision_tree"

    def __init__(self, filename):
        self.filename = filename
        self.loaded_data_file = arff.loadarff(f'./data/{self.filename}')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def clean_data(self):
        df = pd.DataFrame(self.loaded_data_file[0])
        Y_data = df.iloc[:, -1].values

        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(Y_data)
        X_copy = df.iloc[:, :-1].copy()
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_copy)
        new_X = imputer.transform(X_copy)

        ## features selection
        sel = VarianceThreshold(1)
        sel.fit(new_X)
        selected_new_X = sel.transform(new_X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(selected_new_X, y, test_size=0.15, random_state=42)

    def save_data_set(self):
        smell_name = self.filename.split('.')[0].replace('-', '_')
        with open(f"./train_data/{self.model_type}/decision_tree_{smell_name}_train_data.pkl", "wb") as file1:
            pickle.dump([self.X_train, self.y_train], file1)

        with open(f"./test_data/{self.model_type}/decision_tree_{smell_name}_test_data.pkl", "wb") as file2:
            pickle.dump([self.X_test, self.y_test], file2)


    def train_model(self):
        depths = np.arange(10, 50)
        num_leafs = [1, 5, 10, 20, 50, 100, 150]
        param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': depths, 'min_samples_leaf': num_leafs}
        new_tree_clf = DecisionTreeClassifier()
        print(f"Start grid searching for decision tree model with data set: {self.filename}...")
        grid_search = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring="f1", return_train_score=True)

        grid_search.fit(self.X_train, self.y_train)
        print(f"Finish training decision tree model with data set: {self.filename}\n")

        best_model = grid_search.best_estimator_
        smell_name = self.filename.split('.')[0].replace('-', '_')

        pickle.dump(best_model, open(f'./trained_models/{self.model_type}/{smell_name}_decision_tree.sav', 'wb'))



