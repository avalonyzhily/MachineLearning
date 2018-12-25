import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import Imputer


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing/"
HOUSING_URL = DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"


def fetch_housing_data(house_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(house_url,tgz_path)
    housing_tgz =  tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
        return DataFrame Pandas Obj
    """
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data,test_ratio):
    """
    创建测试集,但是要求数据集永远不变
    且该方法每运行一次,测试集就会随机生成一次
    :param data:
    :param test_ratio:
    :return:
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == "__main__":
    # fetch_housing_data()
    housing = load_housing_data(HOUSING_PATH)
    # view top 5 rows
    # print(housing.head())
    # column values range
    # print(housing["ocean_proximity"].value_counts())
    # every column describe
    # print(housing.describe())
    # 纯随机取样
    # housing_train, housing_test = train_test_split(housing, test_size=0.2, random_state=42)
    # 分层采样
    housing["income_cate"] = np.ceil(housing["median_income"]/1.5)
    housing["income_cate"].where(housing["income_cate"] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing,housing["income_cate"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for s in (strat_train_set,strat_test_set):
        s.drop(["income_cate"], axis=1, inplace=True)

    # 创建探索集,使用复制,避免损伤训练集
    housing_discovery = strat_train_set.copy()

    housing_discovery.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)

    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing_backup = strat_train_set.drop("median_house_value", axis=1)

    housing_labels = strat_train_set["median_house_value"].copy()

    # # 数据清洗
    # housing_backup.dropna(subset=["total_bedrooms"])  # 该属性有数据缺失, 直接去掉对应的街区(记录)
    #
    # housing_backup.drop("total_bedrooms", axis=1)  # 去掉该属性
    #
    # median = housing_backup["total_bedrooms"].median()  # 取中位数
    # housing_backup["total_bedrooms"].fillna(median)  # 填充缺失数据

    # 处理缺失值的简单方法, 策略是中位数
    imputer = Imputer(strategy="median")
    # 中位数只能处理数值,因为要去掉文本属性
    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)

    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
