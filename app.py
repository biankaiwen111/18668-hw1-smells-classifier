import os
import pickle

from sklearn.metrics import accuracy_score, f1_score


def run(args):
    all_models_files = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    all_models = list(map(lambda filename: filename.split('.')[0], all_models_files))

    all_smells_files = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    all_smells = list(map(lambda filename: filename.split('.')[0].replace("-", "_"), all_smells_files))
    print(all_models, all_smells)
    model_name = args[0]
    if model_name not in all_models:
        print(f"Model {model_name} does not exist")
        return

    required_smells = args[1:]
    for required_smell in required_smells:
        if required_smell not in all_smells:
            print(f"Unknown smell: {required_smell}")
            return
    run_test_set_result(model_name, required_smells)

def run_test_set_result(model_name, smells):
    print("{0:^16}".format("Smell") + "|" + "{0:^16}".format("Accuracy") + "|" + "{0:^16}".format("F1-score"))
    for smell in smells:
        loaded_model = pickle.load(open(f'./trained_models/{model_name}/{smell}_{model_name}.sav', 'rb'))
        with open(f"./test_data/{model_name}/{model_name}_{smell}_test_data.pkl", "rb") as test_data:
            X_test, y_test = pickle.load(test_data)
            accuracy = str(accuracy_score(y_test, loaded_model.predict(X_test)) * 100)[:5] + "%"
            f1 = str(f1_score(y_test, loaded_model.predict(X_test)) * 100)[:5] + "%"
            print("{0:^16}".format(f"{smell}") + "|" + "{0:^16}".format(f"{accuracy}") + "|" + "{0:^16}".format(f"{f1}"))

def help():
    print("help")

def list_models():
    print('We have following models: (use command "run <model_name> <smells...>" to run the model)')
    all_models = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    for model in all_models:
        print(model.split('.')[0])

def compare(args):
    if len(args) > 2:
        print("You can compare only one model or smell per time")
    all_models_files = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    all_models = list(map(lambda filename: filename.split('.')[0], all_models_files))

    all_smells_files = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    all_smells = list(map(lambda filename: filename.split('.')[0].replace("-", "_"), all_smells_files))

    model_name = args[0]
    if model_name not in all_models:
        print(f"Model {model_name} does not exist")
        return

    required_smell = args[1]
    if required_smell not in all_smells:
        print(f"Unknown smell: {required_smell}")
        return
    compare_test_and_training(model_name, required_smell)

def compare_test_and_training(model_name, smell_name):
    print("{0:^16}".format(f"{smell_name}") + "|" + "{0:^16}".format("Accuracy") + "|" + "{0:^16}".format("F1-score"))
    loaded_model = pickle.load(open(f'./trained_models/{model_name}/{smell_name}_{model_name}.sav', 'rb'))
    with open(f"./train_data/{model_name}/{model_name}_{smell_name}_train_data.pkl", "rb") as test_data:
        X_train, y_train = pickle.load(test_data)
        accuracy_train_set = str(accuracy_score(y_train, loaded_model.predict(X_train)) * 100)[:5] + "%"
        f1_train_set = str(f1_score(y_train, loaded_model.predict(X_train)) * 100)[:5] + "%"
        print("{0:^16}".format("Training set") + "|" + "{0:^16}".format(f"{accuracy_train_set}") + "|" + "{0:^16}".format(f"{f1_train_set}"))
    with open(f"./test_data/{model_name}/{model_name}_{smell_name}_test_data.pkl", "rb") as test_data:
        X_test, y_test = pickle.load(test_data)
        accuracy_test_set = str(accuracy_score(y_test, loaded_model.predict(X_test)) * 100)[:5] + "%"
        f1_test_set = str(f1_score(y_test, loaded_model.predict(X_test)) * 100)[:5] + "%"
        print("{0:^16}".format("Test set") + "|" + "{0:^16}".format(f"{accuracy_test_set}") + "|" + "{0:^16}".format(f"{f1_test_set}"))


if __name__ == "__main__":
    while True:
        try:
            command = input(">> ")
            splitted_command = command.split()

            if len(splitted_command) == 0:
                continue
            elif splitted_command[0] == 'quit':
                break
            elif splitted_command[0] == 'list':
                list_models()
            elif splitted_command[0] == "run":
                run(splitted_command[1:])
            elif splitted_command[0] == "compare":
                compare(splitted_command[1:])
            else:
                print("Unknown command, please try again!")
                print("Enter <help> for usage menu")

            print("your command is: ", splitted_command)
        except Exception as error:
            print(f"Error: {error}")
            break