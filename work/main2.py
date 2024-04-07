import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import shap


def work():
    # label 原标题为：指数偏移值
    data = pd.read_excel("delete_empty_lines2.xls", header=0)
    # 训练数据与测试数据 4:1
    # mask = np.random.rand(len(data)) < 0.8
    # mask = np.random.rand(len(data)) < 1
    # train = data[mask]
    # test = data[~mask]
    # 选中excel中的多列
    # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    selected_column_indices = list(range(6, 25))
    # xgb_train = xgb.DMatrix(train.iloc[:, selected_column_indices], label=train.label)
    # xgb_test = xgb.DMatrix(test.iloc[:, selected_column_indices], label=test.label)

    cols = ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
    model.fit(data[cols], data.label.values)

    # params = {
    #     # "objective": "reg:linear",
    #     "objective": "reg:squarederror",
    #     "booster": "gbtree",
    #     "eta": 0.1,
    #     "min_child_weight": 1,
    #     "max_depth": 16
    # }
    # num_round = 30
    # evals = [(xgb_train, 'train'), (xgb_test, 'test')]
    # evals = [(xgb_train, 'train'), (xgb_train, 'test')]
    # model = xgb.train(params, xgb_train, num_round, evals=evals)
    # model.save_model('model2.json')
    # bst = xgb.Booster()
    # bst.load_model("model2.json")
    # pred = bst.predict(xgb_test)

    pred = model.predict(data[cols])
    print("======== origin label & prediction ========")
    print(f'label index --- label value -- predication')
    i = 0
    for index, value in data.label.items():
        print(f'\t{index} \t\t\t\t{round(value, 3)} \t{pred[i]}')
        i = i + 1


    # print("======== prediction ========")
    # print(pred)
    # print("======== train.label ========")
    # print(type(train.label))
    # print(train.label)
    # print("======== test.label ========")
    # print(test.label)
    print("======== importance ========")
    # importance = model.get_fscore()
    # importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # print(df)
    # df['fscore'] = df['fscore'] / df['fscore'].sum()
    # print(df)

    # ok
    # 获取feature importance
    # plt.figure(figsize=(15, 5))
    # plt.bar(range(len(cols)), model.feature_importances_)
    # plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=14)
    # plt.title('Feature importance', fontsize=14)
    # plt.grid(axis="both", linestyle='-.', alpha=0.3)
    # plt.show()

    # model是在第1节中训练的模型
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[cols])
    print(shap_values.shape)

    # 单个样本的SHAP值
    # 比如我们挑选数据集中的第30位
    # j = 30
    # player_explainer = pd.DataFrame()
    # player_explainer['feature'] = cols
    # player_explainer['feature_value'] = data[cols].iloc[j].values
    # player_explainer['shap_value'] = shap_values[j]

    # ok
    # 单个值的shap分析，存为图片
    # j = 10
    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j], matplotlib=True, show=True)
    # plt.savefig('force_plot.jpg')

    # ok 数据效果不大好，参考 https://zhuanlan.zhihu.com/p/64799119  3.2节
    # 对特征的总体分析
    # shap.summary_plot(shap_values, data[cols])

    # 可以把一个特征对目标变量影响程度的绝对值的均值作为这个特征的重要性。
    # shap.summary_plot(shap_values, data[cols], plot_type="bar")

    # 数据效果不太好
    # 3.3 部分依赖图Partial Dependence Plot
    # 第一个参数备选 ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    # shap.dependence_plot('G', shap_values, data[cols], interaction_index=None, show=True)

    # 数据效果不太好
    # 3.4 对多个变量的交互进行分析
    # shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
    # shap.summary_plot(shap_interaction_values, data[cols], max_display=4)

    # 数据效果不太好
    # 我们也可以用dependence_plot描绘两个变量交互下变量对目标值的影响。
    # 第一个参数备选 ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    shap.dependence_plot('G', shap_values, data[cols], interaction_index='H', show=True)


if __name__ == '__main__':
    work()
