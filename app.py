import os
import pickle

from sklearn.metrics import accuracy_score, f1_score

from train_models import ini_folder_structure, train_models


def run(args):
    all_models_files = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    all_models = list(map(lambda filename: filename.split('.')[0], all_models_files))

    all_smells_files = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    all_smells = list(map(lambda filename: filename.split('.')[0].replace("-", "_"), all_smells_files))
    if len(args) == 0:
        print("Please specify a model. Enter <help> for usage menu")
        return
    model_name = args[0]
    if model_name not in all_models:
        print(f"Model {model_name} does not exist. Enter <list-models> to list all available models")
        return

    required_smells = args[1:]
    if len(required_smells) == 0:
        print("Please specify a least one type of smell. Enter <help> for usage menu")
        return
    for required_smell in required_smells:
        if required_smell not in all_smells:
            print(f"Unknown smell: {required_smell}. Enter <list-smells> to list all available smells")
            return
    run_test_set_result(model_name, required_smells)


def run_test_set_result(model_name, smells):
    print("{0:^16}".format("Smell") + "|" + "{0:^16}".format("Accuracy") + "|" + "{0:^16}".format("F1-score"))
    for smell in smells:
        loaded_model = pickle.load(open(f'./trained_models/{model_name}/{smell}_{model_name}.sav', 'rb'))
        with open(f"./test_data/{model_name}/{model_name}_{smell}_test_data.pkl", "rb") as test_data:
            x_test, y_test = pickle.load(test_data)
            accuracy = str(accuracy_score(y_test, loaded_model.predict(x_test)) * 100)[:5] + "%"
            f1 = str(f1_score(y_test, loaded_model.predict(x_test)) * 100)[:5] + "%"
            print(
                "{0:^16}".format(f"{smell}") + "|" + "{0:^16}".format(f"{accuracy}") + "|" + "{0:^16}".format(f"{f1}"))


def usage_menu():
    print("Commands:")
    print("{0:<40}".format("quit"), "quit the application")
    print("{0:<40}".format("list-models"), "list all available models")
    print("{0:<40}".format("list-smells"), "list all available smells")
    print("{0:<40}".format("run <model> [smell1, smell2...]"), "run model <model> for every smell's test dataset")
    print("{0:<40}".format("compare <model> <smell>"), "compare <model>'s test and training sets for the <smell>")
    print("")


def list_models():
    print('We have following models: (use command <help> for usage menu)')
    all_models = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    for model in all_models:
        print(model.split('.')[0])


def list_smells():
    print('We have following smells: (use command <help> for usage menu)')
    all_smells_files = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    all_smells = list(map(lambda filename: filename.split('.')[0].replace("-", "_"), all_smells_files))
    for smell in all_smells:
        print(smell)


def compare(args):
    if len(args) > 2:
        print("You can compare only one model or smell per time. Enter <help> for usage menu")
        return
    if len(args) != 2:
        print("Please specify the model or smell. Enter <help> for usage menu")
        return
    all_models_files = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    all_models = list(map(lambda filename: filename.split('.')[0], all_models_files))

    all_smells_files = list(filter(lambda filename: filename.endswith(".arff"), os.listdir('data')))
    all_smells = list(map(lambda filename: filename.split('.')[0].replace("-", "_"), all_smells_files))

    model_name = args[0]
    if model_name not in all_models:
        print(f"Model {model_name} does not exist. Enter <list-models> to list all available models")
        return

    required_smell = args[1]
    if required_smell not in all_smells:
        print(f"Unknown smell: {required_smell}. Enter <list-smells> to list all available smells")
        return
    compare_test_and_training(model_name, required_smell)


def compare_test_and_training(model_name, smell_name):
    print("{0:^16}".format(f"{smell_name}") + "|" + "{0:^16}".format("Accuracy") + "|" + "{0:^16}".format("F1-score"))
    loaded_model = pickle.load(open(f'./trained_models/{model_name}/{smell_name}_{model_name}.sav', 'rb'))
    with open(f"./train_data/{model_name}/{model_name}_{smell_name}_train_data.pkl", "rb") as test_data:
        x_train, y_train = pickle.load(test_data)
        accuracy_train_set = str(accuracy_score(y_train, loaded_model.predict(x_train)) * 100)[:5] + "%"
        f1_train_set = str(f1_score(y_train, loaded_model.predict(x_train)) * 100)[:5] + "%"
        print(
            "{0:^16}".format("Training set") + "|" + "{0:^16}".format(f"{accuracy_train_set}") + "|" + "{0:^16}".format(
                f"{f1_train_set}"))
    with open(f"./test_data/{model_name}/{model_name}_{smell_name}_test_data.pkl", "rb") as test_data:
        x_test, y_test = pickle.load(test_data)
        accuracy_test_set = str(accuracy_score(y_test, loaded_model.predict(x_test)) * 100)[:5] + "%"
        f1_test_set = str(f1_score(y_test, loaded_model.predict(x_test)) * 100)[:5] + "%"
        print("{0:^16}".format("Test set") + "|" + "{0:^16}".format(f"{accuracy_test_set}") + "|" + "{0:^16}".format(
            f"{f1_test_set}"))


def verify_folder_structure():
    all_models = list(filter(lambda filename: filename.endswith(".py"), os.listdir('models')))
    for model in all_models:
        model_name = model.split('.')[0]
        if not os.path.exists(f"test_data/{model_name}") or not os.path.exists(
                f"train_data/{model_name}") or not os.path.exists(f"trained_models/{model_name}"):
            return False
    return True


if __name__ == "__main__":
    if not verify_folder_structure():
        print("Use the application for the first time: start training models...")
        ini_folder_structure()
        train_models()
        print("Finished training! Start application!")
    while True:
        try:
            command = input(">> ")
            split_command = command.split()

            if len(split_command) == 0:
                continue
            elif split_command[0] == 'quit':
                break
            elif split_command[0] == 'list-models':
                list_models()
            elif split_command[0] == "list-smells":
                list_smells()
            elif split_command[0] == "run":
                run(split_command[1:])
            elif split_command[0] == "compare":
                compare(split_command[1:])
            elif split_command[0] == "help":
                usage_menu()
            else:
                print("Unknown command, please try again!")
                print("Enter <help> for usage menu")
        except Exception as error:
            print(f"Error: {error}")
            break
