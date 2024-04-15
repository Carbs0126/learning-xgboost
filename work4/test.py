import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import shap
import os
import datetime
import seaborn as sns

# 参考 https://zhuanlan.zhihu.com/p/64799119


def work():
    store_dir_path = get_current_store_dir()
    force_plot_dir_path = get_current_force_plot_dir(store_dir_path)
    # label 原标题为：指数偏移值
    data = pd.read_excel("test.xls", header=0)
    # 训练数据与测试数据 4:1
    # mask = np.random.rand(len(data)) < 0.8
    # mask = np.random.rand(len(data)) < 1
    # train = data[mask]
    # test = data[~mask]
    # 选中excel中的多列
    # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    selected_column_indices = list(range(0, 8))
    # xgb_train = xgb.DMatrix(train.iloc[:, selected_column_indices], label=train.label)
    # xgb_test = xgb.DMatrix(test.iloc[:, selected_column_indices], label=test.label)

    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
    model.fit(data[cols], data.label.values)

    # 使用所有原始数据进行训练，然后使用这些原始数据进行预测
    pred = model.predict(data[cols])

    file_name_predication = os.path.join(store_dir_path, "predication.txt")
    file_predication = open(file_name_predication, 'w', encoding='utf-8')
    print("======== origin label & prediction ========")
    print("label index --- label value -- predication")
    file_predication.write("label index --- label value -- predication\n")
    for index, value in data.label.items():
        line = f'\t{index} \t\t\t\t{round(value, 3)} \t{pred[index]}'
        print(line)
        file_predication.write(line + "\n")
    file_predication.close()

    print("======== importance ========")
    file_name_importance_txt = os.path.join(store_dir_path, "importance.txt")
    file_importance_txt = open(file_name_importance_txt, 'w', encoding='utf-8')

    # ok 绘importance图并存储
    # 获取feature importance
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(cols)), model.feature_importances_)
    plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=14)
    plt.title('Feature importance', fontsize=14)
    plt.grid(axis="both", linestyle='-.', alpha=0.3)
    plt.savefig(os.path.join(store_dir_path, "plot_importance.png"), dpi=300)
    # plt.show()
    plt.close()

    # 将importance图写入文件
    file_importance_txt.write("column name -- importance\n")
    print("column name -- importance\n")
    for index, value in enumerate(cols):
        line = f'\t{value} \t\t\t\t{model.feature_importances_[index]} \t'
        print(line)
        file_importance_txt.write(line + "\n")
    file_importance_txt.close()

    # 使用SHAP分析， model是在第1节中训练的模型
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[cols])

    # 将 SHAP 值转换成 Pandas DataFrame
    shap_df = pd.DataFrame(shap_values, columns=data[cols].columns)

    # 计算 SHAP 值的相关性矩阵
    correlation_matrix = shap_df.corr()

    # 每一行的数据都生成一个图片，可能会比较耗时
    max_pic = 10
    shap.initjs()
    for index, value in data.label.items():
        if index > max_pic:
            break
        # shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j], matplotlib=True, show=True)
        shap.force_plot(explainer.expected_value, shap_values[index], data[cols].iloc[index], matplotlib=True,
                        show=False)
        fig = plt.savefig(os.path.join(force_plot_dir_path, f"force_plot_{index}.jpg"), dpi=300)
        plt.close(fig)
        print(f"force_plot {index + 1}/{data.label.size} finish.")

    # ok 数据效果不大好，参考 https://zhuanlan.zhihu.com/p/64799119  3.2节
    # 对特征的总体分析
    shap.summary_plot(shap_values, data[cols], show=False)
    fig = plt.savefig(os.path.join(store_dir_path, "plot_summary_dot.png"), dpi=300, format='png')
    plt.close(fig)

    # 可以把一个特征对目标变量影响程度的绝对值的均值作为这个特征的重要性。
    shap.summary_plot(shap_values, data[cols], plot_type="bar", show=False)
    fig = plt.savefig(os.path.join(store_dir_path, "plot_summary_bar.png"), dpi=300, format='png')
    plt.close(fig)

    # 数据效果不太好
    # 3.3 部分依赖图Partial Dependence Plot
    # 第一个参数备选 ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    shap.dependence_plot('G', shap_values, data[cols], interaction_index=None, show=False)
    fig = plt.savefig(os.path.join(store_dir_path, "plot_dependence.png"), dpi=300, format='png')
    plt.close(fig)

    # 数据效果不太好
    # 3.4 对多个变量的交互进行分析
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
    # max_display 可调，最大是列数，即 len(cols)
    shap.summary_plot(shap_interaction_values, data[cols], max_display=4, show=False)
    fig = plt.savefig(os.path.join(store_dir_path, "plot_summary_interaction.png"), dpi=300, format='png')
    plt.close(fig)

    # 数据效果不太好
    # 我们也可以用dependence_plot描绘两个变量交互下变量对目标值的影响。
    # 第一个参数备选 ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    dependence_column_name_0 = 'A'
    dependence_column_name_1 = 'F'
    shap.dependence_plot(dependence_column_name_0, shap_values, data[cols], interaction_index=dependence_column_name_1,
                         show=False)
    fig = plt.savefig(
        os.path.join(store_dir_path, f"plot_dependence_{dependence_column_name_0}_{dependence_column_name_1}.png"),
        dpi=300, format='png')
    plt.close(fig)

    # 创建热力图
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("SHAP Values Correlation Matrix")
    fig = plt.savefig(os.path.join(store_dir_path, "cool_warm.png"), dpi=300, format='png')
    plt.close(fig)
    # plt.show()


def get_current_store_dir():
    project_path = os.path.dirname(os.path.realpath(__file__))
    time_dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dir_path = os.path.join(project_path, time_dir_name)
    os.makedirs(dir_path)
    return dir_path


def get_current_force_plot_dir(store_dir):
    force_plot_dir_path = os.path.join(store_dir, "force_plot")
    os.makedirs(force_plot_dir_path)
    return force_plot_dir_path


if __name__ == '__main__':
    work()
