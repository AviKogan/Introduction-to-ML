import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def add_ones_column_at_start(X: np.array) -> np.array:
    if X.ndim == 1:
        X = X.reshape((1, X.shape[0]))

    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    return X


def calc_score(samp_num: int, predictions: np.array, y: np.array) -> dict():
    res = dict()
    P = N = TP = FP = TN = FN = 0

    for i in range(samp_num):
        if y[i] == 1 and predictions[i] == 1:
            TP += 1
            P += 1
        elif y[i] == -1 and predictions[i] == -1:
            TN += 1
            N += 1
        elif y[i] == 1 and predictions[i] == -1:
            FN += 1
            P += 1
        else:  # y[i] == -1 and predictions[i] == 1:
            FP += 1
            N += 1

    res["num_samples"] = samp_num
    res["error"] = (FP + FN) / samp_num
    res["accuracy"] = (TP + TN) / samp_num
    res["FPR"] = FP / N
    res["TPR"] = TP / P
    res["precision"] = TP / (TP + FP)
    res["specificity"] = TN / N

    return res


class Perceptron:
    model = None

    # iterate over the samples and check if there is sample that
    # misclassified and return its index, if all classes classified correctly
    # return -1.
    def __get_misclassified_index(self, X: np.array, y: np.array) -> int:
        for i in range(X.shape[0]):
            if y[i] * np.matmul(X[i, :], self.model) <= 0:
                return i

        return -1

    def __sign(self, num):
        if num < 0:
            return -1
        return 1

    def fit(self, X: np.array, y: np.array):
        X = add_ones_column_at_start(X)
        self.model = np.zeros((X.shape[1], 1))  # model size according to X
        # columns
        while True:
            i = self.__get_misclassified_index(X, y)
            if i == -1:
                # all samples classified correctly
                return self

            self.model = self.model + (y[i] * X[i, :]).reshape(-1, 1)

    def predict(self, X: np.array) -> np.array:
        X = add_ones_column_at_start(X)
        predictions = np.empty((X.shape[0], 1))
        for i in range(X.shape[0]):
            predictions[i] = self.__sign(np.matmul(X[i, :], self.model))

        return predictions

    def score(self, X: np.array, y: np.array) -> dict:
        predictions = self.predict(X)
        return calc_score(X.shape[0], predictions, y)


class LDA:
    # model will store in the keys "Ber", "M-", "M+" and "Sigma"
    # the params of the model
    # "Ber" - the bernoulli estimator of y
    # "M-" - the expectation estimator of x/(y=-1)
    # "M+" - the expectation estimator of x/(y=+1)
    # "Sigma" - the Variance-Covariance matrix estimator of both x/y
    model = {}
    __Ber = "Ber"
    __M_neg = "M-"
    __M_pos = "M+"
    __Sigma = "Sigma"

    def fit(self, X: np.array, y: np.array):
        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        pos_y_num = 0
        x_of_y_pos_sum = np.zeros((1, X.shape[1]))
        x_of_y_neg_sum = np.zeros((1, X.shape[1]))

        for i in range(X.shape[0]):
            if y[i] == 1:
                x_of_y_pos_sum += X[i, :]
                pos_y_num += 1
            else:
                x_of_y_neg_sum += X[i, :]

        self.model[self.__Ber] = pos_y_num / X.shape[0]
        self.model[self.__M_pos] = x_of_y_pos_sum / pos_y_num
        self.model[self.__M_neg] = x_of_y_neg_sum \
                                   / (X.shape[0] - pos_y_num)

        sigma = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            if y[i] == 1:
                cur = (X[i, :] - self.model[self.__M_pos])
                sigma += np.matmul(np.transpose(cur), cur)
            else:
                cur = (X[i, :] - self.model[self.__M_neg])
                sigma += np.matmul(np.transpose(cur), cur)

        self.model[self.__Sigma] = (1 / X.shape[0]) * sigma
        self.model[self.__M_pos] = self.model[self.__M_pos].reshape(-1, 1)
        self.model[self.__M_neg] = self.model[self.__M_neg].reshape(-1, 1)
        return self

    def predict(self, X: np.array) -> np.array:
        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        predictions = np.empty((X.shape[0], 1))

        sigma_inv = np.linalg.inv(self.model[self.__Sigma])
        m_pos_Sig_m_pos = -0.5 * np.transpose(self.model[self.__M_pos]) @ \
                          sigma_inv @ self.model[self.__M_pos]
        m_neg_Sig_m_neg = -0.5 * np.transpose(self.model[self.__M_pos]) @ \
                          sigma_inv @ self.model[self.__M_pos]

        ln_Pr_pos = np.log(self.model[self.__Ber])
        ln_Pr_neg = np.log(1 - self.model[self.__Ber])

        for i in range(X.shape[0]):
            x_sigma = X[i, :] @ sigma_inv

            delta_plus = x_sigma @ self.model[self.__M_pos] + \
                         m_pos_Sig_m_pos + ln_Pr_pos

            delta_neg = x_sigma @ self.model[self.__M_neg] + \
                        m_neg_Sig_m_neg + ln_Pr_neg

            predictions[i] = 1 if delta_plus > delta_neg else -1

        return predictions

    def score(self, X: np.array, y: np.array) -> dict:
        predictions = self.predict(X)
        return calc_score(X.shape[0], predictions, y)


class SVM:

    model = None

    def fit(self, X: np.array, y: np.array):
        self.model = SVC(C=1e10, kernel='linear').fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def score(self, X: np.array, y: np.array) -> dict:
        predictions = self.predict(X)
        return calc_score(X.shape[0], predictions, y)


class Logistic:

    model = None

    def fit(self, X: np.array, y: np.array):
        self.model = LogisticRegression(solver='liblinear').fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def score(self, X: np.array, y: np.array) -> dict:
        predictions = self.predict(X)
        return calc_score(X.shape[0], predictions, y)


class DecisionTree:

    model = None

    def fit(self, X: np.array, y: np.array):
        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        #choose the tree max_depth to be the Features number
        self.model = DecisionTreeClassifier(max_depth=X.shape[1]).fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def score(self, X: np.array, y: np.array) -> dict:
        predictions = self.predict(X)
        return calc_score(X.shape[0], predictions, y)


if __name__ == '__main__':
    pass


