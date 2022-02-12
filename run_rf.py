import pickle
from sklearn.metrics import accuracy_score

loaded_model = pickle.load(open('./trained_models/random_forest/long_method_random_forest.sav', 'rb'))
with open("./test_data/random_forest/random_forest_long_method_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/random_forest/random_forest_long_method_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/random_forest/god_class_random_forest.sav', 'rb'))
with open("./test_data/random_forest/random_forest_god_class_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/random_forest/random_forest_god_class_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/random_forest/data_class_random_forest.sav', 'rb'))
with open("./test_data/random_forest/random_forest_data_class_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/random_forest/random_forest_data_class_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))

loaded_model = pickle.load(open('./trained_models/random_forest/feature_envy_random_forest.sav', 'rb'))
with open("./test_data/random_forest/random_forest_feature_envy_test_data.pkl", "rb") as test_data:
    X_test, y_test = pickle.load(test_data)
with open("./train_data/random_forest/random_forest_feature_envy_train_data.pkl", "rb") as train_data:
    X_train, y_train = pickle.load(train_data)
print("\nThe test accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_test, loaded_model.predict(X_test))))
print("\nThe train accuracy for loaded model is: ")
print("DT loaded " + str(accuracy_score(y_train, loaded_model.predict(X_train))))