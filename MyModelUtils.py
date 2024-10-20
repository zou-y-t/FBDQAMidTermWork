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

    def __init__(self, df, security_code=None):
        """
        Args:
            df: 股票数据
            security_code(Optional): 股票代码 

        """
        self.df = df
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.start_date = self.df["date"][0]
        self.end_date = self.df["date"][-1]
        self.security = security_code
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
        """
        删除ST股票/退市股票
        Args:
            None
        Return:
            Boolean(True or False)
            True: 存在异常
            False: 无异常
        
        """
        if self.security is None:
            return False
        
        for name in self.security:
            st_df = get_extras(
                "is_st", name, start_date=self.start_date, end_date=self.end_date
            )
            if not st_df["is_st"] == False.all():
                # 某一天被ST了
                return True
            
        return False

    def pause(self):
        """对于涨跌停处理"""

    def ignore(self):
        """缺失的数据"""
        self.df = self.df.dropna()

    def clean(self):
        """
        进行数据清理
        """
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


class WeightRestore:
    """
    对于某一支股票在一段时间内进行复权
    Args:
        security:股票代码
        start_date:起始日期
        end_date:终止日期
    """

    def __init__(self, security, start_date, end_date):
        self.security = security
        self.start_date = start_date
        self.end_date = end_date

    def getResult(self):
        """
        获取复权结束后的结果
        Returns:
            pd.DataFrame: columns: "close","open","date","is_st"
        """
        bars = get_bars(self.security, self.start_date, self.end_date, unit="1d")
        close = bars["close"]
        fq_factors = get_adj_factors(
            self.security, self.start_date, self.end_date, fq="pre"
        )

        fq_close = self.calc_fq(close, fq_factors)
        is_st = get_extras("is_st", self.security, self.start_date, self.end_date)
        limit = get_limit_price(self.security, self.start_date, self.end_date)
        dfs = [self.df, fq_close, is_st]
        if type(limit) == pd.DataFrame and len(limit) > 0:
            dfs.append(limit)
        else:
            print(f"没有查询到涨跌停数据：标的{self.security}, 长期停牌或已经退市")
        df = pd.concat(dfs, sort=True, axis=1)
        df.dropna(inplace=True)

        # 删除st
        df = df[df["is_st"] == False]

        # 删除涨跌停
        df = df[(df["close"] != df["up"]) & (df["close"] != df["down"])]
        return df

    def cal_fq(close, fq_factors):
        """
        获取复权[前复权]数据(在getResult中调用,无需在外部调用)
        Args:
            param close: 收盘价序列
            param fq_factors:  DataFrame类型: 复权因子序列
        
        Returns:
            Series类型, 复权后的收盘价序列
        """

        if type(fq_factors) == pd.DataFrame and len(fq_factors) > 0:
            # 复权价 = 累计前复权因子 * 不复权价
            factor = fq_factors["factor"]
            df = pd.concat([close, factor], sort=True, axis=1)
            # 只有除息日才有复权因子。除息日不需要复权, 所以把除息日的复权因子与除息日前一个交易日对齐。
            df["factor"] = df["factor"].shift(-1)
            # 使用除息日前一个交易日的复权因子, 填充该交易日之前的复权因子
            df = df.fillna(method="bfill")
            # 使用1填充,最近一个除息日、以及该除息日之后每个交易日的复权因子
            df = df.fillna(value=1)
            # 默认价格是np.float64类型，复权因子是float类型，计算前先做数据类型转换
            df["fq_close"] = df["close"] * (df["factor"].apply(np.float64))
            return df["fq_close"]
        else:
            return close.rename("fq_close")


class SpecialData:
    """
    用于处理和获取特殊数据
    波动率高/成交高的 股票
    重要日期的实现
    """

    def __init__(self, df, date):
        """
        Args:
            param1 df : columns:"volume","name","date","price"
            param2 date : time(date)
        """
        self.df = df
        self.date = pd.to_datetime(date)

    def getHiVOL(self, standard=None, percentage=None):
        """
        获取高成交量的股票
        Args:
            param1(int Optional) standard:不输入standard,默认在df中排序,前5% 
            param2(float Optional) percentage:输入percentage,按照前percentage筛选

        Returns:
            return(DataFrame) sorted_df
            description:
            返回 VOL由高到低的 df
            columns:'volume'
            columns:'name'

        """
        filtered_df = self.df.copy()
        if percentage is None:
            if standard is None:
                filtered_df = filtered_df[filtered_df["volume"] > standard]
            else:
                threshold = self.df["volume"].quantile(0.95)
                filtered_df = filtered_df[filtered_df["volume"] > threshold]
        else:
            threshold = self.df["volume"].quantile(1-percentage)
            filtered_df = filtered_df[filtered_df["volume"] > threshold]

        sorted_df = filtered_df.sort_values(by="volume", ascending=False)

        return sorted_df
    
    def getDays(self, security_code=None, r1, r7=None, r8):
        """
        获取特殊日期的模型结果
        Args:
            param1: security_code (str, optional): 证券代码
            param2: r1 (float): 参数1
            param3: r7 (float, optional): 参数7
            param4: r8 (float): 参数8

        Returns:
            pd.DataFrame: columns:date、alpha、beta、R2、err、security_code(Optional)
        """
        results = []
        # 输入df - 返回损失率
        for date in self.dates:
            df = self.df.copy()
            date_data = df.loc[(df["date"] == date)]
            if r7 is None:
                result = Model1(date_data, r8=r8, r1=r1)
                predictions = result.predict(r1=r1)
            else:
                result = Model2(date_data, r1=r1, r7=r7, r8=r8)
                predictions = result.predict(r1=r1,r7=r7)

        result.fit()
        beta = result.getBeta()
        alpha = result.getAlpha()
        R2 = result.getR2()


        actuals = r8 
        error = (abs(predictions - actuals)).mean()  # 误差

        metric = {"date": date, "alpha": alpha, "beta": beta, "R2": R2, "security_code": security_code, "error": error}
        if security_code is None:
            metric["security_code"] = security_code
        
        results.append(metric)

        return pd.DataFrame(results)
