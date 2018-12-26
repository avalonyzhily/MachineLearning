import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing/"
HOUSING_URL = DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"


def fetch_housing_data(house_url=HOUSING_URL,housing_path=HOUSING_PATH):
    """
        获取数据
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(house_url,tgz_path)
    housing_tgz =  tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
        使用pandas加载数据
    """
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
# 获取数据csv
fetch_housing_data(HOUSING_URL,HOUSING_PATH)
# 加载数据得到DataFrame
housing = load_housing_data(HOUSING_PATH)

"""
    目标：使用已有的普查数据建立房价模型,预测任何街区的房价中位数
    使用数据：加州1990年普查数据(经纬度、房龄中位数、总房间数、总卧室数、人口数、户数、收入中位数、房价中位数、离海岸距离)
    
    分析：有标签的样本——房价中位数——>监督学习,需要预测值——>回归任务(不需要预测值则不需要回归,只需要分类)
          没有连续数据进入系统,不需要快速适应,数据量不大可以放进内存,使用批量学习

    性能指标：回归任务经典指标
            RMSE(均方根误差)——测量系统预测误差的标准差,通常符合高斯分布(68-95-99.7,σ就是RMSE,68%在1σ,95%在2σ,99.7%在3σ)
                公式：RMSE(X,h) x是所有特征值的向量(不包含标签),y是标签(即输出值),X包含是数据集中所有实例的特征值,每一行都是一个实例的特征值的转置(列转行,每个特征一列)
                      h是预测函数,y = h(x)
            
            MAE(平均绝对偏差)——异常值较多时,使用该值更好

            测量预测值和目标值两个向量距离的方法/范数：RMSE——欧几里得范数,MAE——曼哈顿范数,K阶闵氏范数(0阶为汉明范数,高阶为切比雪夫范数)
            范数的指数越高,就越关注大的值而忽略小的值

    确认需要的是实际的值,而不是将值分类(回归任务变成分类任务)
"""


"""
    pandas 数据统计包,需要数据pandas的使用
    pandas.plotting.scatter_matrix函数,绘制每个属性与其他属性的关系图
    read_csv() 读取数据文件,返回DataFrame对象
    DataFrame对象：
        head()方法——数据集的前五行
        info()方法——查看数据描述,总行数、每个属性的类型、非空值数量.etc
        describe()方法——数值属性的概括(数量,均值,标准差,最小值,最大值,)
        loc(index)方法——选中index行
        copy()方法——创建副本
        corr()方法——查看每对属性的标准相关系数(结果是一个属性两两成对矩阵)
    DataFrame[key]返回对应项的数据集
        values_counts()方法——统计该项有哪些类别,每个类别有多少数量
    
    柱状图有利于快速了解数据类型和分布

    特征值预处理(放大/缩放,上下限),但是要注意预测值(标签)的上下限是否符合要求：
        对于作了限制的标签可以选择重新收集合适的标签,或是从训练集移除

    柱状图的分布比较靠某一边,需要对数据做一些变换使其更符合正态分布
"""

"""
    数据拆分成训练集和测试集,训练集还可以继续拆分成训练集和验证集(多个训练集和验证集交叉验证)

    随机采样的测试集偏差相比分层(特定指标比例更具有代表性)采样的测试集更大,分层的标准则需要离散化数据并限制分类数

    生成测试集后,放在一边,用于测试训练好的模型
"""

# 根据收入中位数分类来分层采样,增加一个临时属性,除以1.5限制分类数量(连续数据离散化)
housing["income_cat"] = np.ceil(housing["median_income"]/1.5) #收入中位数离散化
housing["income_cat"].where(housing["income_cat"] < 5,5.0,inplace=True) #分类

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 采样完毕去掉临时属性
for s in (strat_train_set, strat_test_set):
    s.drop(["income_cat"],axis=1,inplace=True)

"""
    研究训练集,如果训练集较大,则可以在分拆出一个探索集来寻找数据规律,反之则可以直接在训练全集上操作,创建副本以避免损伤训练集
"""
# 创建训练集副本
housing_copy = strat_train_set.copy()

"""    
    matplotlib.pyplot show() 可以用来输出图形
"""
# 直接用经纬度来描点,突出数据分布的点
# housing_copy.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)


# 还可以增加可视化参数,探索规律
# housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s = housing["population"]/100, label="population", 
#     c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar=True)

# plt.legend()
# plt.show()

"""
    查找关联性,通过corr()方法来获得属性之间的相关系数,相关系数是线性的关系
"""
# corr_matrix = housing_copy.corr()

# 查看每个属性与房价中位数的相关系数
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

"""
    也可是使用pandas的scatter_matrix函数绘制两两属性之间的关系图(属性可以指定需要关注的)
"""
# 设置关注的属性
# attributes = ["median_house_value", "median_income", "total_rooms",
# "housing_median_age"]
# 绘制图
# scatter_matrix(housing_copy[attributes], figsize=(12, 8))
# plt.show()

"""
    本案例中与目标属性(房价中位数)相关性最高的是收入中位数,即根据收入中位数来预测房价中位数最有希望
    1. 有清晰的(向上)趋势,且数据不分散
    2. 上限的直线与最高价(设置了上限)吻合,并且在下面的一切区间也有一些直线(对应的街区可能是需要去除的,防止算法重复巧合)
"""
# 放大并独立观察收入中位数与房价中位数的关系图(与目标属性最具相关性的属性)
# housing_copy.plot(kind="scatter", x="median_income",y="median_house_value",
# alpha=0.1)

# plt.show()

"""
    除了已经提供的属性,还可以利用既有属性的关系,来组合出一些新属性来分析
    尝试多种属性组合
    探索规律
"""
# # 每户拥有房间数
# housing_copy["rooms_per_household"] = housing_copy["total_rooms"]/housing_copy["households"]
# # 总卧室与总房间数的比例
# housing_copy["bedrooms_per_rooms"] = housing_copy["total_bedrooms"]/housing_copy["total_rooms"]
# # 每户人口数
# housing_copy["population_per_household"] = housing_copy["population"]/housing_copy["households"]

# corr_matrix = housing_copy.corr()

# print(corr_matrix["median_house_value"].sort_values(ascending=False))

"""
    为机器学习算法准备数据
    准备干净的训练集(再次复制训练集)
    将预测量和标签(目标值)分开
    对预测量和标签应用不同的转换
"""
# inplace 默认false,返回操作后的新DataFrame不影响原来的
housing_copy_new = strat_train_set.drop("median_house_value",axis=1)
# 复制房价中位数的副本
housing_labels = strat_train_set["median_house_value"].copy()

"""
    数据清洗
    创建一些函数来处理特征缺失问题：
    DataFrame有现成的方法
    1. 去掉对应记录(去行) dropna()
    2. 去掉整个属性(去列) drop()
    3. 进行赋值(0、均值、中位数等有代表性的值) fillna()
"""
