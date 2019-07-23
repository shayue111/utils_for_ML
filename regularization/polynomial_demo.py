from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def make_data():
    """生成一维数据并且返回，y=sin(4x) + noise"""
    np.random.seed(1)
    X = np.sort(np.random.rand(30))
    y = np.sin(4 * X) + np.random.randn(30) * 0.3
    return X, y


def plot_data(X, y):
    """ 画点 """
    plt.plot(X, y, 'k.')
    plt.xlabel('X')
    plt.ylabel('y')
    # plt.show()


def get_polynomial_feature(origin_features, deg):
    """
    生成多项式数据
    :param origin_features: 多维数组，本例中shape为(n,1)
    :param deg: 扩展维度
    :return: 扩展后的np.array
    """

    polynomial = PolynomialFeatures(
        degree=deg,
        include_bias=False
    )
    polynomial_features = polynomial.fit_transform(origin_features)
    return polynomial_features


if __name__ == '__main__':
    # 生成数据
    features, target = make_data()
    features = features.reshape(-1, 1)
    # 在图上显示
    plot_data(features, target)

    for i in [1]:
        poly_data = get_polynomial_feature(features, i)
        model = LinearRegression()
        model.fit(poly_data, target)
        # print(f"degree - {i}:", model.coef_)  # 查看模型训练得到的参数

        # 插值处理画图平滑曲线
        x = features.squeeze()                                # 生成插值的数据只能是一维
        pred_y = model.predict(poly_data)

        new_x = np.arange(x.min(), x.max(), 0.0002)           # 插值范围不能超过原数据的最小最大值
        func = interpolate.interp1d(x, pred_y, kind='cubic')  # kind方法：zero、slinear、quadratic、cubic
        new_y = func(new_x)

        # 画图
        plt.plot(new_x, new_y, label='degree' + str(i))

    plt.legend()
    plt.axis([0, 1, -1.5, 2])      # 设置横轴纵轴长度
    plt.show()
