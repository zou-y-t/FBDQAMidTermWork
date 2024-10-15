import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Model1:
    """
    Model1: y=alpha+beta*x
    """

    def __init__(self, r8, r1):
        self.r8 = np.array(r8)
        self.r1 = np.array(r1)
        self.beta = 0
        self.alpha = 0
        self.predictR8 = np.array([])

    def fit(self):
        # 计算斜率beta
        self.beta = np.cov(self.r1, self.r8)[0][1] / np.var(self.r1)
        # 计算截距alpha
        self.alpha = np.mean(self.r8) - self.beta * np.mean(self.r1)
        # 计算预测值
        self.predictR8 = self.alpha + self.beta * self.r1
        # 计算R2
        self.R2 = 1 - (self.r8 - self.predictR8).dot(self.r8 - self.predictR8) / np.sum(
            (self.r8 - np.mean(self.r8)) ** 2
        )

    def getBeta(self):
        return self.beta

    def getAlpha(self):
        return self.alpha

    def getR2(self):
        return self.R2

    def predict(self, r):
        return self.alpha + self.beta * r

    def plot(self):
        plt.scatter(self.r1, self.r8)
        plt.plot(self.r1, self.alpha + self.beta * self.r1, color="red")
        plt.xlabel("r1")
        plt.ylabel("r8")
        plt.title("Scatter plot with regression line")
        plt.show()


class Model2:
    """
    Model2: y=alpha+beta1*x1+beta2*x2
    """

    def __init__(self, r8, r1, r7):
        self.r8 = np.array(r8)
        self.r1 = np.array(r1)
        self.r7 = np.array(r7)
        self.beta_r1 = 0
        self.beta_r7 = 0
        self.alpha = 0
        self.predictR8 = np.array([])
        self.R2 = 0

    def fit(self):
        # 计算斜率beta
        self.beta_r1 = np.cov(self.r1, self.r8)[0][1] / np.var(self.r1)
        self.beta_r7 = np.cov(self.r7, self.r8)[0][1] / np.var(self.r7)
        # 计算截距alpha
        self.alpha = (
            np.mean(self.r8)
            - self.beta_r1 * np.mean(self.r1)
            - self.beta_r7 * np.mean(self.r7)
        )
        # 计算预测值
        self.predictR8 = self.alpha + self.beta_r1 * self.r1 + self.beta_r7 * self.r7
        # 计算R2
        self.R2 = 1 - (self.r8 - self.predictR8).dot(self.r8 - self.predictR8) / np.sum(
            (self.r8 - np.mean(self.r8)) ** 2
        )

    def getBeta_r1(self):
        return self.beta_r1

    def getBeta_r7(self):
        return self.beta_r7

    def getAlpha(self):
        return self.alpha

    def getR2(self):
        return self.R2

    def predict(self, r1, r7):
        return self.alpha + self.beta_r1 * r1 + self.beta_r7 * r7

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.r1, self.r7, self.r8)
        x1 = np.linspace(min(self.r1), max(self.r1), 100)
        x2 = np.linspace(min(self.r7), max(self.r7), 100)
        x1, x2 = np.meshgrid(x1, x2)
        y = self.alpha + self.beta_r1 * x1 + self.beta_r7 * x2
        ax.plot_surface(x1, x2, y, cmap="rainbow")
        ax.set_xlabel("r1")
        ax.set_ylabel("r7")
        ax.set_zlabel("r8")
        plt.show()


class DataCleaner:
    """
    DataCleaner: df是2022/4/1-2022/6/30的分钟数据，这个类取了半小时的数据，计算了r
    """

    half_hour_time = [
        # 半小时时间
        "10:00:00",
        "10:30:00",
        "11:00:00",
        "11:30:00",
        "13:30:00",
        "14:00:00",
        "14:30:00",
        "15:00:00",
    ]

    def __init__(self, df, security_code):
        """df是数据,security_code是股票代码"""
        self.df = df
        self.code = security_code
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.start_date = self.df["date"][0]
        self.end_date = self.df["date"][-1]
        self.r1, self.r2, self.r3, self.r4, self.r5, self.r6, self.r7, self.r8 = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.r = [
            self.r1,
            self.r2,
            self.r3,
            self.r4,
            self.r5,
            self.r6,
            self.r7,
            self.r8,
        ]

    def st_quit(self):
        """删除ST股票/退市股票"""

    def pause(self):
        """对于涨跌停处理"""

    def ignore(self):
        """缺失的数据"""

    def clean(self):
        # 去除半小时结点的数据
        self.df = self.df[
            self.df["date"].dt.time.astype(str).isin(self.half_hour_time)
        ].copy()
        # 计算r
        self.df.loc[:, "r"] = self.df["close"] / self.df["close"].shift(1) - 1
        # 去除2022/4/1的数据（缺损了第一个数据点r1）
        self.df = self.df[self.df["date"].dt.date.astype(str) != "2022-04-01"]
        # 将r依次存入r1,r2,r3,r4,r5,r6,r7,r8
        for i in range(len(self.df)):
            self.r[i % 8].append(self.df["r"].iloc[i])

    def getR(self):
        return self.r

    def getR1(self):
        return self.r1

    def getR2(self):
        return self.r2

    def getR3(self):
        return self.r3

    def getR4(self):
        return self.r4

    def getR5(self):
        return self.r5

    def getR6(self):
        return self.r6

    def getR7(self):
        return self.r7

    def getR8(self):
        return self.r8


class SpecialData:
    """
    用于处理和获取特殊数据
    波动率高/成交高的 股票
    重要日期的实现
    """

    def __init__(self, df):
        self.df = df
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.start_date = self.df["date"][0]
        self.end_date = self.df["date"][-1]
        # self.special_days = [] 目前未确定哪些特殊日期

    def getHiVOL(self, standard=None):
        """
        获取高成交量的股票
        如果不输入standard,默认在df中排序，前5%
        如果输入standard就按照比standard高筛选
        """
        filtered_df = self.df.copy()
        if standard is None:
            filtered_df = filtered_df[filtered_df["volume"] > standard]
        else:
            threshold = self.df["volume"].quantile(0.95)
            filtered_df = filtered_df[filtered_df["volume"] > threshold]

        sorted_df = filtered_df.sort_values(by="volume", ascending=False)

        return sorted_df

    def getHiVot(self, standard=None):
        """
        获取高波动率的股票
        如果不输入standard,默认在df中排序，前5%
        如果输入standard就按照比standard高筛选
        ** 如何获取波动率？ 日内数据？ **
        """

        return self.df

    def getDays(self, Model=None, security_code=None):
        """
        获取特殊日期的模型结果
        需要输入模型，证券代码
        然后对于指定证券进行分析
        返回目前未确定
        """
        return None
