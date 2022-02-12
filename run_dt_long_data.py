import pickle
from sklearn.metrics import accuracy_score

loaded_model = pickle.load(open('./trained_models/decision_tree/long_method_decision_tree.sav', 'rb'))
with open("./test_data/decision_tree/decision_tree_long_method_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/decision_tree/decision_tree_long_method_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/decision_tree/god_class_decision_tree.sav', 'rb'))
with open("./test_data/decision_tree/decision_tree_god_class_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/decision_tree/decision_tree_god_class_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/decision_tree/data_class_decision_tree.sav', 'rb'))
with open("./test_data/decision_tree/decision_tree_data_class_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/decision_tree/decision_tree_data_class_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/decision_tree/feature_envy_decision_tree.sav', 'rb'))
with open("./test_data/decision_tree/decision_tree_feature_envy_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/decision_tree/decision_tree_feature_envy_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))