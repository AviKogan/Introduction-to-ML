import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import models as ex_models

def label_point(x) -> float:
    return 1 if 0.3 * x[0] - 0.5 * x[1] + 0.1 > 0 else -1



#Q8
def draw_points(m: int):
    mu, cov = np.array([0, 0]), np.array([[1., 0.], [0., 1.]])
    x, y = [], []
    for _ in range(m):
        cur = np.random.multivariate_normal(mu, cov)
        x.append(cur)
        y.append(label_point(cur))

    return np.array(x), np.array(y)


def Q9():
    samp_m = [5, 10, 15, 25, 70]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[rf"$\text{{models with m = {m}}}$"
                                        for m in samp_m])
    symbols = np.array(["circle", "x"])
    xx = np.linspace(-5, 5)
    first = samp_m[0]
    for i in range(len(samp_m)):
        show_leg = True if samp_m[i] == first else False

        x, y = draw_points(samp_m[i])

        f_coef = [0.1, 0.3, -0.5]
        a = -f_coef[1] / f_coef[2]
        y_f = a * xx - (f_coef[0]) / f_coef[2]

        perceptron_coefs = ex_models.Perceptron().fit(x, y).model
        a = -perceptron_coefs[1] / perceptron_coefs[2]
        y_per = a * xx - (perceptron_coefs[0]) / perceptron_coefs[2]

        svc = ex_models.SVM().fit(x, y).model
        w = svc.coef_[0]
        a = -w[0] / w[1]
        y_svm = a * xx - (svc.intercept_[0]) / w[1]

        fig.add_traces([
            go.Scatter(x=x[:,0], y=x[:,1], mode="markers", showlegend=False,
                       marker=dict(color=y, symbol=symbols[y],
                                   colorscale=["orange", "blue"],
                                   line=dict(color="black", width=1))),

            go.Scatter(x=xx, y=y_f, mode='lines', name="f", legendgroup="f",
                       showlegend=show_leg,
                       line=dict(color="yellow")),

            go.Scatter(x=xx, y=y_per, mode='lines', name="perceptron",
                       legendgroup="perceptron",
                       showlegend=show_leg,
                       line=dict(color="red")),

            go.Scatter(x=xx, y=y_svm, mode='lines', name="SVM",
                       legendgroup="SVM",
                       showlegend=show_leg,
                       line=dict(color="blue"))],

            rows=(i // 3) + 1, cols=(i % 3) + 1)


    fig.show()


def Q10():
    samp_m = [5, 10, 15, 25, 70]

    k = 1000
    repeats = 500
    perc_acc_list = []
    svm_acc_list = []
    lda_acc_list = []

    for m in samp_m:
        perc_acc = .0
        svm_acc = .0
        lda_acc = .0
        for _ in range(repeats):
            while True:
                x, y = draw_points(m)
                if 1 in y and -1 in y:
                    break
            test_x, test_y = draw_points(k)

            perc = ex_models.Perceptron().fit(x, y)
            svm = ex_models.SVM().fit(x, y)
            lda = ex_models.LDA().fit(x, y)

            perc_acc += perc.score(test_x, test_y)["accuracy"]
            svm_acc += svm.score(test_x, test_y)["accuracy"]
            lda_acc += lda.score(test_x, test_y)["accuracy"]

        perc_acc_list.append(perc_acc / repeats)
        svm_acc_list.append(svm_acc / repeats)
        lda_acc_list.append(lda_acc / repeats)

    plt.plot(samp_m, perc_acc_list, label="Perception")
    plt.plot(samp_m, svm_acc_list, label="SVM")
    plt.plot(samp_m, lda_acc_list, label="LDA")
    plt.xlabel('m')
    plt.ylabel('mean accuracy over 500 repeats')
    plt.title('Models mean accuracy as function of m,\n where m is the train '
              'sample size')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Q9()
    # Q10()
    pass