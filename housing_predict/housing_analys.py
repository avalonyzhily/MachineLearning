import os
import tarfile
from six.moves import urllib
import pandas as pd
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
    read_csv() 读取数据文件,返回DataFrame对象
    DataFrame对象：
        head()方法——数据集的前五行
        info()方法——查看数据描述,总行数、每个属性的类型、非空值数量.etc
        describe()方法——数值属性的概括(数量,均值,标准差,最小值,最大值,)
        loc(index)方法——选中index行
    DataFrame[key]返回对应项的数据集
        values_counts()方法——统计该项有哪些类别,每个类别有多少数量

    柱状图有利于快速了解数据类型和分布

    特征值预处理(放大/缩放,上下限),但是要注意预测值(标签)的上下限是否符合要求：
        对于作了限制的标签可以选择重新收集合适的标签,或是从训练集移除

    柱状图的分布比较靠某一边,需要对数据做一些变换使其更符合正态分布
"""

"""
    数据拆分成训练集和测试集,训练集还可以继续拆分成训练集和验证集(多个训练集和验证集交叉验证)
"""
fetch_housing_data(HOUSING_URL,HOUSING_PATH)
data = load_housing_data(HOUSING_PATH)
print(data)