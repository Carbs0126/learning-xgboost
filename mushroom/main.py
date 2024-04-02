import xgboost as xgb
import numpy as np
import pandas as pd


# https://github.com/dmlc/xgboost/tree/master/demo/data
# print(xgb.libpath)

def mushroom():
    # 数据读取 这里需要主动添加 format=libsvm
    xgb_train = xgb.DMatrix('agaricus.txt.train?format=libsvm')
    xgb_test = xgb.DMatrix('agaricus.txt.test?format=libsvm')

    # 定义模型训练参数
    params = {"objective": "binary:logistic", "booster": "gbtree", "max_depth": 3}

    # 训练轮数
    num_round = 10

    # 训练过程中实时输出评估结果
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]

    evals = [(xgb_train, 'train'), (xgb_test, 'test')]
    # 模型训练
    # model = xgb.train(params, xgb_train, num_round, watchlist)
    model = xgb.train(params, xgb_train, num_round, evals=evals)

    # 预测结果
    preds = model.predict(xgb_test)
    print("========predict result========")
    # preds 类型：<class 'numpy.ndarray'>
    print(preds)
    # 将 preds 做变换，不影响原来的值
    preds_transformed = np.where(preds < 0.5, 0, 1)
    print("========transformed result========")
    print(preds_transformed)
    print("========statistic result========")
    test_data_frame = pd.read_csv("agaricus.txt.test", header=None, sep='\s+')
    error_rate = np.sum(preds_transformed != test_data_frame[0]) / test_data_frame.shape[0]
    print("测试集错误总数: {}".format(np.sum(preds_transformed != test_data_frame[0])))
    print("测试集错误率(softmax): {}".format(error_rate))

    print("========importance result========")
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=lambda x:x[1], reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print(df)
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)


if __name__ == '__main__':
    mushroom()
