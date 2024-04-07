import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance


def work():
    # label 原标题为：指数偏移值
    data = pd.read_excel("delete_empty_lines.xls", header=0)
    # 训练数据与测试数据 4:1
    # mask = np.random.rand(len(data)) < 0.8
    mask = np.random.rand(len(data)) < 1
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
        "max_depth": 16
    }
    num_round = 30
    evals = [(xgb_train, 'train'), (xgb_train, 'test')]
    model = xgb.train(params, xgb_train, num_round, evals=evals)
    model.save_model('model.json')
    bst = xgb.Booster()
    bst.load_model("model.json")
    # pred = bst.predict(xgb_test)
    pred = bst.predict(xgb_train)
    print("======== origin label & prediction ========")
    print(f'label index --- label value -- predication')
    i = 0
    for index, value in train.label.items():
        print(f'\t{index} \t\t\t\t{round(value, 3)} \t{pred[i]}')
        i = i + 1

    print("======== importance ========")
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    print(df)
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)

    # plt.figure(figsize=(15, 5))
    # plt.bar(range(len(importance)), importance)
    # plt.xticks(range(len(importance)), importance, rotation=-45, fontsize=14)
    # plt.title('Feature importance', fontsize=14)
    # plt.show()



    # feature_names = ['Feature_{}'.format(i) for i in range(len(importance))]
    # scores = importance

    # 可以按降序排序后再绘制
    # sorted_indices = np.argsort(scores)
    # sorted_feature_names = [feature_names[i] for i in sorted_indices]
    # sorted_scores = [scores[i] for i in sorted_indices]

    # 绘制条形图
    # plt.figure(figsize=(10, 6))
    # plt.bar(sorted_feature_names, sorted_scores)
    # plt.xlabel('特征名')
    # plt.ylabel('特征重要性得分')
    # plt.title('特征重要性分析')
    # plt.xticks(rotation=90)  # 若特征名过长，可旋转标签
    # plt.show()

    # OK
    plot_importance(model)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.grid(axis="both", linestyle='-.', alpha=0.3)
    plt.show()


if __name__ == '__main__':
    work()
