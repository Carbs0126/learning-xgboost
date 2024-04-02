import xgboost as xgb
import pandas as pd
import numpy as np


def concrete():
    data = pd.read_excel("concrete_data.xls")
    # print(data.head(10))
    data.rename(columns={"Concrete compressive strength(MPa, megapascals) ": 'label'}, inplace=True)
    # print(data.head(10))
    # 训练数据与测试数据 4:1
    mask = np.random.rand(len(data)) < 0.8
    train = data[mask]
    test = data[~mask]
    print(type(test))

    # 应该是8 才对
    xgb_train = xgb.DMatrix(train.iloc[:, :7], label=train.label)
    xgb_test = xgb.DMatrix(test.iloc[:, :7], label=test.label)
    # print(train.iloc[:, :8].head(2))
    # 模型训练，reg:linear 默认评估采用 RMSE
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eta": 0.1,
        "min_child_weight": 1,
        "max_depth": 5
    }
    num_round = 50
    evals = [(xgb_train, 'train'), (xgb_test, 'test')]
    model = xgb.train(params, xgb_train, num_round, evals=evals)
    # model.save_model("model.xgb")
    # bst = xgb.Booster()
    # bst.load_model("model.xgb")
    model.save_model('model.json')
    bst = xgb.Booster()
    bst.load_model("model.json")
    pred = bst.predict(xgb_test)
    print(pred)
    # print(type(pred))
    # print(type(test.label))
    arr_delta = pred - test.label
    print(arr_delta)
    # test_data_frame = pd.read_csv("agaricus.txt.test", header=None, sep='\s+')
    # error_rate = np.sum(preds_transformed != test_data_frame[0]) / test_data_frame.shape[0]
    # print("测试集错误总数: {}".format(np.sum(preds_transformed != test_data_frame[0])))
    # print("测试集错误率(softmax): {}".format(error_rate))


if __name__ == '__main__':
    concrete()
