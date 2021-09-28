"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""


import ex4_tools as ex4_t
import numpy as np
import matplotlib.pyplot as plt

# y, h_X are to numpy vectors, d is the distributes of h_x
def calc_epsilon(D, y, h_x):

    not_eq = np.where(h_x != y)[0]
    eps = np.sum(D[not_eq])

    return eps


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.full((m), 1/m)  # init distribution
        # lp = LineProfiler()
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            h_x = self.h[t].predict(X)
            # h_x is the result of current h_t on each row of X
            # eps_t = calc_epsilon(D, y, h_x, m)
            eps_t = calc_epsilon(D, y, h_x)
            w_t = 0.5 * np.log((1/eps_t) - 1)
            self.w[t] = w_t

            D = np.multiply(D, np.exp(-1 * w_t * np.multiply(y, h_x)))
            D = D / np.sum(D)

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        total = 0
        for t in range(max_t):
            total += self.w[t] * self.h[t].predict(X)

        return np.sign(total)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        misclassification_num = float(np.where(y != y_hat)[0].shape[0])
        return misclassification_num / X.shape[0]


def Q13(X_train, y_train, X_test, y_test, Ab, T):
    test_error, train_error = [], []
    T_range = np.arange(1, T)
    for i in T_range:
        test_error.append(Ab.error(X_test, y_test, i))
        train_error.append(Ab.error(X_train, y_train, i))

    plt.plot(T_range, test_error, label="Test error")
    plt.plot(T_range, train_error, label="Train_error")
    plt.xlabel('T')
    plt.ylabel('Error rate')
    plt.title('Adaboost train/test error as function of T')
    plt.legend()
    plt.show()

def Q14(X_test, y_test, Ab):
    T_range = np.array([5, 10, 50, 100, 200, 500])

    D = np.full((200), 1 / 200)
    fig, axarr = plt.subplots(nrows=2, ncols=3)
    for i in range(6):

        plt.sca(axarr[int(i/3), i%3])
        ex4_t.decision_boundaries(Ab, X_test, y_test, T_range[i], D)

    fig.tight_layout(pad=3.0)
    plt.show()


def Q15(X_train, y_train, Ab):
    T_range = np.arange(2, 500)
    min_error = Ab.error(X_train, y_train, 1)
    min_train_error_i = 0
    for t in T_range:
        cur = Ab.error(X_train, y_train, t)
        if cur < min_error:
            min_train_error_i = t
            min_error = cur

    D = np.full((5000), 1 / 5000)

    plt.subplot()
    ex4_t.decision_boundaries(Ab, X_train, y_train, min_train_error_i, D)
    plt.title(f"decision boundary for t={min_train_error_i} that minimize "
              f"train error,\n  train error = {min_error}")
    plt.show()


def Q16(D_T, X_train, y_train, Ab):
    plt.subplot()
    ex4_t.decision_boundaries(Ab, X_train, y_train, 500, D_T)
    plt.title(f"Decision boundary for t=500,\n "
              f"the train points with weights of t=500")
    plt.show()

def Q13_16(noise=0):
    m = 5000
    m_test = 200
    T = 500
    X_train, y_train = ex4_t.generate_data(m, noise)
    X_test, y_test = ex4_t.generate_data(m_test, noise)
    Ab = AdaBoost(ex4_t.DecisionStump, T)
    D_T = Ab.train(X_train, y_train)
    Q13(X_train, y_train, X_test, y_test, Ab, T)
    Q14(X_test, y_test, Ab)
    Q15(X_train, y_train, Ab)
    D_T = D_T / np.max(D_T) * 10
    Q16(D_T, X_train, y_train, Ab)

def Q17():
    noise_ratio = [0.01, 0.04]
    for n in noise_ratio:
        Q13_16(n)

if __name__ == '__main__':
    np.random.seed(100)
    Q13_16()
    Q17()


