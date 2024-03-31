import xgboost as xgb


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
    print(preds)


if __name__ == '__main__':
    mushroom()
