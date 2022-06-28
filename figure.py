import matplotlib.pyplot as plt

if __name__ == "__main__":
    X=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    data = [21.31, 11.38, 6.05, 4.05, 1.80, 1.17, 0.93, 0.51, 0.18]
    data2 = [16.09, 7.32, 3.26, 1.06, 0.67, 0.18, 0.14, 0.12, 0.09]
    plt.plot(X, data, label="filter+Apriori-closed")
    plt.plot(X, data2, label="ExAnte+Apriori-closed")
    plt.xlabel('Threshold of support')
    plt.ylabel("execution time of algorithms")
    plt.legend()
    plt.show()