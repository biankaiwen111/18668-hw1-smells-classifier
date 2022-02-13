from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
import os
import shutil

from models.svc_rbf import SVCrbf


def ini_folder_structure():
    model_types = os.listdir('models')

    if os.path.exists("./test_data"):
        shutil.rmtree("./test_data")
    for model_type in model_types:
        model_name = model_type.split(".")[0]
        os.makedirs(f"test_data/{model_name}")

    if os.path.exists("./train_data"):
        shutil.rmtree("./train_data")
    for model_type in model_types:
        model_name = model_type.split(".")[0]
        os.makedirs(f"train_data/{model_name}")

    if os.path.exists("./trained_models"):
        shutil.rmtree("./trained_models")
    for model_type in model_types:
        model_name = model_type.split(".")[0]
        os.makedirs(f"trained_models/{model_name}")


def train_models():
    data_sets = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    for data_set in data_sets:
        decision_tree = DecisionTree(data_set)
        decision_tree.clean_data()
        decision_tree.save_data_set()
        decision_tree.train_model()

        random_forest = RandomForest(data_set)
        random_forest.clean_data()
        random_forest.save_data_set()
        random_forest.train_model()

        svc_rbf = SVCrbf(data_set)
        svc_rbf.clean_data()
        svc_rbf.save_data_set()
        svc_rbf.train_model()


if __name__ == '__main__':
    try:
        ini_folder_structure()
        train_models()
        print("Finished training!")
    except Exception as error:
        print(f"Error: {error}")
