import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import timeit

def filter_date(data):
    x, y = [], []
    for l in data:
        if np.logical_or(l[0] == 0, (l[0] == 1)):
            y.append(l[0])
            x.append(l[1:])

    return x, y

def load_data(base_path):
    train_data = np.loadtxt(base_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(base_path + "mnist_test.csv",
                           delimiter=",")
    return train_data, test_data


def Q12(x, y):
    one_left_to_print = zero_left_to_print = 3
    i = 0
    while one_left_to_print != 0 or zero_left_to_print != 0:
        if one_left_to_print > 0 and y[i] == 1:
            one_left_to_print -= 1
            img = x[i].reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()

        elif zero_left_to_print > 0 and y[i] == 0:
            zero_left_to_print -= 1
            img = x[i].reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()

        i += 1


#Q13
def rearrange_data(X):
    X = np.array(X)
    return X.reshape(X.shape[0], 784)


def Q14(x_train, y_train, x_test, y_test):
    samp_m = [50, 100, 300, 500]

    models = [LogisticRegression(),
              SVC(C=1),
              KNeighborsClassifier(n_neighbors=4),
              DecisionTreeClassifier(max_depth=10)]

    models_accuracy = [[], [], [], []]
    models_time = []
    repeats = 50
    model_num = 0
    for model in models:
        model_time = 0
        for m in samp_m:
            cur_acc = 0
            for _ in range(repeats):
                x, y = [], []
                while True:
                    for i in np.random.randint(low=0, high=len(x_train),
                                               size=m):
                        y.append(y_train[i])
                        x.append(x_train[i])

                    if 1 in y and 0 in y:
                        break

                start = timeit.default_timer()

                model.fit(x, y)
                cur_acc += model.score(x_test, y_test)

                stop = timeit.default_timer()

                model_time += (stop - start)

            models_accuracy[model_num].append(cur_acc / repeats)

        models_time.append(model_time)
        model_num += 1

    plt.plot(samp_m, models_accuracy[0], label="Logistic Regression")
    plt.plot(samp_m, models_accuracy[1], label="Soft-SVM")
    plt.plot(samp_m, models_accuracy[2], label="4-Neighbors Classifier")
    plt.plot(samp_m, models_accuracy[3], label="DecisionTreeClassifier")
    plt.xlabel('m')
    plt.ylabel('mean accuracy over 50 repeats')
    plt.title('Models mean accuracy as function of m,\n where m is the train '
              'sample size')
    plt.legend()
    plt.show()

    print("times respectively: LogisticRegression, SVC, KNeighborsClassifier, "
          "DecisionTreeClassifier ", end="")
    print(models_time)


if __name__ == '__main__':

    # data_path = "/Users/avikogan/Desktop/"
    # train_data, test_data = load_data(data_path)
    #
    # x_train, y_train = filter_date(train_data)
    # x_test, y_test = filter_date(test_data)
    # Q12(x_train, y_train)
    # Q14(x_train, y_train, x_test, y_test)
    pass
