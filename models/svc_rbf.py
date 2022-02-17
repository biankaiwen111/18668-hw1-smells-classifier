import pandas as pd
import pickle
from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


class SVCrbf:
    model_type = "svc_rbf"

    def __init__(self, filename):
        self.filename = filename
        self.loaded_data_file = arff.loadarff(f'./data/{self.filename}')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def clean_data(self):
        df = pd.DataFrame(self.loaded_data_file[0])
        y_data = df.iloc[:, -1].values

        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(y_data)
        x_copy = df.iloc[:, :-1].copy()
        imputer = SimpleImputer(strategy="median")
        imputer.fit(x_copy)
        new_x = imputer.transform(x_copy)

        # features selection
        sel = VarianceThreshold(1)
        sel.fit(new_x)
        selected_new_x = sel.transform(new_x)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(selected_new_x, y, test_size=0.15,
                                                                                random_state=42)

    def save_data_set(self):
        smell_name = self.filename.split('.')[0].replace('-', '_')
        with open(f"./train_data/{self.model_type}/svc_rbf_{smell_name}_train_data.pkl", "wb") as file1:
            pickle.dump([self.X_train, self.y_train], file1)

        with open(f"./test_data/{self.model_type}/svc_rbf_{smell_name}_test_data.pkl", "wb") as file2:
            pickle.dump([self.X_test, self.y_test], file2)

    def train_model(self):
        c = [200, 250, 280, 310, 500, 820, 850, 900, 950, 1000]
        gammas = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
        param_grid = {'C': c, 'kernel': ['rbf'], 'gamma': gammas}

        svc = SVC()
        print(f"Start grid searching for svc(rbf kernel) model with data set: {self.filename}...")
        k_fold = StratifiedKFold(n_splits=10)
        grid_search = GridSearchCV(svc, param_grid, cv=k_fold, scoring="accuracy", return_train_score=True)

        grid_search.fit(self.X_train, self.y_train)
        print(f"Finish training svc(rbf kernel) model with data set: {self.filename}\n")

        best_model = grid_search.best_estimator_
        smell_name = self.filename.split('.')[0].replace('-', '_')
        pickle.dump(best_model, open(f'./trained_models/{self.model_type}/{smell_name}_svc_rbf.sav', 'wb'))
