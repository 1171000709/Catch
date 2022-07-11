import pandas as pd


def load_data(samples=1000, classification: bool = True, split_train_size: float = 0.0):


    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split

    if classification:  #
        data_x, data_y = make_classification(n_samples=samples, n_classes=4, n_features=10, n_informative=8)
    else:  #
        data_x, data_y = make_regression(n_samples=samples, n_features=10)

    df_x = pd.DataFrame(data_x, columns=['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', "f_7", "f_8", "f_9", "f_10"])
    df_y = pd.Series(data_y)
    if split_train_size:
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=split_train_size, random_state=0, shuffle=True)
        return x_train, y_train, x_test, y_test
    else:
        return df_x, df_y


if __name__ == '__main__':
    x, y = load_data(samples=10000, classification=False)  #
    train_x, test_x, train_y, test_y = load_data(split_train_size=0.8)  #
