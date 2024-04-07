import xgboost as xgb
import pandas as pd
import numpy as np


def work():
    # label 原标题为：指数偏移值
    data = pd.read_excel("delete_empty_lines.xls", header=0)
    # 训练数据与测试数据 4:1
    mask = np.random.rand(len(data)) < 0.8
    # mask = np.random.rand(len(data)) < 1
    train = data[mask]
    test = data[~mask]
    # 选中excel中的多列
    # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    selected_column_indices = list(range(6, 25))
    xgb_train = xgb.DMatrix(train.iloc[:, selected_column_indices], label=train.label)
    xgb_test = xgb.DMatrix(test.iloc[:, selected_column_indices], label=test.label)

    params = {
        # "objective": "reg:linear",
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "eta": 0.1,
        "min_child_weight": 1,
        "max_depth": 5
    }
    num_round = 20
    evals = [(xgb_train, 'train'), (xgb_test, 'test')]
    # evals = [(xgb_train, 'train'), (xgb_train, 'test')]
    model = xgb.train(params, xgb_train, num_round, evals=evals)
    model.save_model('model.json')
    bst = xgb.Booster()
    bst.load_model("model.json")
    pred = bst.predict(xgb_test)
    # pred = bst.predict(xgb_train)
    print("======== test label & prediction ========")
    print(f'label index --- label value -- predication')
    i = 0
    for index, value in test.label.items():
        print(f'\t{index} \t\t\t\t{round(value, 3)} \t{pred[i]}')
        i = i + 1

    print("======== importance ========")
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print(df)
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)


if __name__ == '__main__':
    work()
