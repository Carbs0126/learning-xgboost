import xgboost as xgb
import pandas as pd
import numpy as np


def seeds():
    # 把index=7的列的值，每个数据都减少1
    data = pd.read_csv("seeds_dataset.txt", header=None, sep='\s+', converters={7: lambda x: int(x) - 1})

    # inplace=True 参数表示在原始数据框架上进行原地修改，而不是返回一个修改后的副本。
    # 因此，这行代码将会直接在原始的 data 数据框架上修改第7列的列名为 'label'。
    data.rename(columns={7: 'label'}, inplace=True)
    # print(data.head(210))

    # 生成一维随机数，并选择小于0.8的数据
    mask = np.random.rand(len(data)) < 0.8

    # print(mask)

    train = data[mask]
    test = data[~mask]
    # 选择 0,1,2,3,4,5,6 这些列，[:7] 等同于 [0:7]
    # print(train.iloc[:, :7])

    # 生成DMatrix
    # 所有的行，前7列
    xgb_train = xgb.DMatrix(train.iloc[:, :7], label=train.label)
    xgb_test = xgb.DMatrix(test.iloc[:, :7], label=test.label)

    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 5,
        'num_class': 3
    }

    evals = [(xgb_train, 'train'), (xgb_test, 'test')]
    num_round = 50
    model = xgb.train(params, xgb_train, num_round, evals=evals)

    # 模型预测
    pred = model.predict(xgb_test)

    # print(type(pred))         # <class 'numpy.ndarray'>
    # print(type(test))         # <class 'pandas.core.frame.DataFrame'>
    # print(type(test.label))   # <class 'pandas.core.series.Series'>
    error_rate = np.sum(pred != test.label) / test.shape[0]
    print("测试集错误率(softmax): {}".format(error_rate))

    print("========importance result========")
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=lambda x:x[1], reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print(df)
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)


if __name__ == '__main__':
    seeds()
