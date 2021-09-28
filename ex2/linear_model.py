import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
:return numpy array represent the given sigma after dagger operation
"""
def get_sigma_dagger_values(sigma: np.array, epsilon) -> np.array:
    for i in range(sigma.shape[0]):
        sigma[i] = 1 / sigma[i] if sigma[i] > epsilon else 0
    return sigma

"""
Adds ones column to create design matrix from the given train data.
"""
def get_design_matrix(data: np.array) -> np.array:
    return np.append(arr=np.ones([data.shape[0], 1]).astype(int),
                     values=data, axis=1)


"""
Q9
:return numpy array with the sigma dagger singular values after filtering 
values that smaller than given epsilon.
@:param x: the design matrix
@:param y: response vector
"""
def fit_linear_regression(x: np.array, y: np.array) -> (np.array, np.array):
    s_of_x = np.linalg.svd(x, compute_uv=False)  # the singular values of the
                                                 # given x matrix

    u, s, v_T = np.linalg.svd(get_design_matrix(x))
    epsilon = np.exp(-6)
    s_dagger_values = get_sigma_dagger_values(s, epsilon)
    S_dagger = np.zeros((v_T.shape[0], u.shape[0]))
    np.fill_diagonal(S_dagger, s_dagger_values)
    w = v_T.transpose() @ S_dagger @ u.transpose() @ y
    return w, s_of_x


"""
Q10
Returns the prediction according to the given design matrix and coefficients.  
@:param x: the design matrix
@:param w: the coefficients
"""
def predict(x: np.array, w: np.array) -> np.array:
    return get_design_matrix(x) @ w


"""
Q11
Calculating the MSE of given response and prediction.  
@:param y: response vector
@:param p: prediction vector
"""
def mse(y: np.array, p: np.array) -> np.array:
    return np.mean((y - p) ** 2)

"""
Q12,Q13
Receive path to the csv file of the house_data, removes samples with 
invalid data or change the invalid data to valid one. finally returns 
response vector that is the 'price' column and the dataframe without the 
'price' column. 
"""
def load_data(path: str):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # dropping the id column because its irrelevance for the linear
    # regression model
    df.drop("id", inplace=True, axis=1)

    # list of the columns there values should be greater than 0
    cols_greater_than_zero = ["price", "bedrooms", "sqft_living",
                                 "sqft_lot", "floors", "sqft_above",
                              "yr_built", "zipcode", "sqft_living15",
                              "sqft_lot15"]

    # list of the columns there values should be greater or equal than 0
    cols_greater_eq_than_zero = ["bathrooms", "yr_renovated", "sqft_basement"]

    # unhandled yet: date, waterfront, view, condition, grade, lat, long

    for col in cols_greater_eq_than_zero:
        df = df[df[col] >= 0]
    for col in cols_greater_than_zero:
        df = df[df[col] > 0]

    # handling: date, waterfront, view, condition, grade
    # no change in: lat, long

    date = pd.to_datetime(df["date"])
    df["date"] = date.map(dt.datetime.toordinal)
    df = df[df.waterfront.isin([0, 1])]
    df = df[df.view.isin(range(0, 5))]
    df = df[df.condition.isin(range(1, 6))]
    df = df[df.grade.isin(range(1, 14))]

    #hot-code the zipcode
    df = pd.get_dummies(df, columns=["zipcode"])

    return df

"""
Q14, plot screw-plot of given singular values. 
"""
def plot_singular_values(s: np.array) -> None:
    s[::-1].sort()  # sort the values in descending order

    singular_value_numbers = np.arange(s.shape[0]) + 1
    plt.plot(singular_value_numbers, s, 'p')
    plt.title('Scree Plot')
    plt.xlabel('Number of the singular value')
    plt.ylabel('Singular value')
    plt.show()

def Q15(path: str) -> None:
    df = load_data(path)
    y_np = df["price"].to_numpy()
    x_np = df.drop("price", axis=1).to_numpy()
    w, singular_values = fit_linear_regression(x_np, y_np)
    plot_singular_values(singular_values)
    # print(singular_values[60:]) # prints the the singular values start
                                  # from the one in the 60 place


def Q16(path: str) -> None:
    df = load_data(path)
    test = df.sample(frac=0.25)
    train = df.drop(test.index)

    y_test = test["price"].to_numpy()
    x_design_test = test.drop("price", axis=1).to_numpy()

    y_train = train["price"].to_numpy()
    x_design_train = train.drop("price", axis=1).to_numpy()

    train_num_samples = train.shape[0]
    mse_res = []

    for i in range(1, 101):
        last_sample = int(train_num_samples * i * 0.01)
        cur_y = y_train[:last_sample]
        cur_x = x_design_train[:last_sample, :]

        w, singular_values = fit_linear_regression(cur_x, cur_y)

        y_hat = predict(x_design_test, w)

        mse_res.append(mse(y_test, y_hat))

    percentage = np.arange(100) + 1

    plt.plot(percentage, mse_res, 'p')
    plt.title('MSE of trained model on p percentage of the train data')
    plt.xlabel('p - %')
    plt.ylabel('MSE')
    plt.show()


def Q17(path: str) -> None:
    df = load_data(path)
    non_categorical = ["bedrooms", "bathrooms", "floors", "view",
                       "sqft_living", "sqft_lot", "sqft_above", "grade",
                       "condition", "yr_built", "yr_renovated", "lat", "long",
                       "sqft_basement", "sqft_living15", "sqft_lot15", "date"]

    prices = df["price"].to_numpy()
    sd_price = np.std(prices)

    for feature in non_categorical:
        cur_as_np = df[feature].to_numpy()

        pearson = (np.cov(prices.transpose(), cur_as_np) /
                   (np.std(cur_as_np) * sd_price))[0][1]

        plt.scatter(df[feature], df["price"])
        plt.annotate(f"\u03C1 = {round(pearson, 3)}", (10, 12),
                     xycoords="figure points", bbox=dict(boxstyle="round", fc="0.8"),)
        plt.title(f'Scatter of price based on {feature}')
        plt.xlabel(f'{feature}')
        plt.ylabel('price')
        plt.show()


if __name__ == '__main__':
    houses_path = "/Users/avikogan/Desktop/kc_house_data.csv"
    Q15(houses_path)
    Q16(houses_path)
    Q17(houses_path)

