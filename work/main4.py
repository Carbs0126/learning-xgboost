import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import shap
import os
import datetime
import seaborn as sns
import matplotlib.colors as mcolors

# 参考 https://zhuanlan.zhihu.com/p/64799119


def work():
    store_dir_path = get_current_store_dir()
    force_plot_dir_path = get_current_force_plot_dir(store_dir_path)
    # label 原标题为：指数偏移值
    # data = pd.read_excel("delete_empty_lines2.xls", header=0)
    data = pd.read_excel("with_empty_lines.xls", header=0)
    # 训练数据与测试数据 4:1
    # mask = np.random.rand(len(data)) < 0.8
    # mask = np.random.rand(len(data)) < 1
    # train = data[mask]
    # test = data[~mask]
    # 选中excel中的多列
    # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # selected_column_indices = list(range(6, 25))
    # xgb_train = xgb.DMatrix(train.iloc[:, selected_column_indices], label=train.label)
    # xgb_test = xgb.DMatrix(test.iloc[:, selected_column_indices], label=test.label)

    cols = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
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
    # without_zero_importance
    cols_zero_importance = []
    for index, value in enumerate(cols):
        importance_v = model.feature_importances_[index]
        if importance_v < 0.0001:
            cols_zero_importance.append(value)
        line = f'\t{value} \t\t\t\t{importance_v} \t'
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
    print("====================")
    print(correlation_matrix.shape)
    print(correlation_matrix.head(20))

    correlation_matrix_without_zero = correlation_matrix.drop(cols_zero_importance, axis=1)
    correlation_matrix_without_zero = correlation_matrix_without_zero.drop(cols_zero_importance, axis=0)
    print("========== correlation_matrix_without_zero ==========")
    print(correlation_matrix_without_zero.shape)
    print(correlation_matrix_without_zero.head(20))
    # correlation_matrix.drop()

    # cols_zero_importance

    # print(cols_zero_importance)

    # shap_values_without_z_i = explainer.shap_values(data[cols_without_z_i])
    # 将 SHAP 值转换成 Pandas DataFrame
    # shap_df_without_z_i = pd.DataFrame(shap_values, columns=data[cols_without_z_i].columns)
    # 计算 SHAP 值的相关性矩阵
    # correlation_matrix_without_z_i = shap_df_without_z_i.corr()

    # 每一行的数据都生成一个图片，可能会比较耗时
    # shap.initjs()
    # for index, value in data.label.items():
    #     # shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j], matplotlib=True, show=True)
    #     shap.force_plot(explainer.expected_value, shap_values[index], data[cols].iloc[index], matplotlib=True,
    #                     show=False)
    #     fig = plt.savefig(os.path.join(force_plot_dir_path, f"force_plot_{index}.jpg"), dpi=300)
    #     plt.close(fig)
    #     print(f"force_plot {index + 1}/{data.label.size} finish.")

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
    plt.figure(figsize=(15, 8))
    col_name_for_shap_value = 'G'
    shap.dependence_plot(col_name_for_shap_value, shap_values, data[cols], interaction_index=None, show=False)
    fig = plt.savefig(os.path.join(store_dir_path, f"plot_dependence_{col_name_for_shap_value}_for_shap_value.png"),
                      dpi=300, format='png')
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
    # 第一个参数备选 ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    dependence_column_name_0 = 'G'
    dependence_column_name_1 = 'H'
    shap.dependence_plot(dependence_column_name_0, shap_values, data[cols], interaction_index=dependence_column_name_1,
                         show=False)
    fig = plt.savefig(
        os.path.join(store_dir_path, f"plot_dependence_{dependence_column_name_0}_{dependence_column_name_1}.png"),
        dpi=300, format='png')
    plt.close(fig)

    # ========================== 创建热力图 ==========================
    plt.figure(figsize=(10, 10))
    # sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    # sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r")
    # sns.heatmap(data=data, cmap="RdBu_r")

    # full_palette = sns.color_palette("viridis", 256)  # 获取完整的viridis颜色列表
    full_palette = sns.color_palette("RdBu_r", 256)

    # 我们可以选择其中的一部分来创建一个新的ListedColormap
    start_index = int(0.2 * len(full_palette))  # 开始位置（20%）
    end_index = int(0.8 * len(full_palette))  # 结束位置（80%）
    reduced_palette = full_palette[start_index:end_index]

    # 创建新的ListedColormap
    cmap_reduced = mcolors.ListedColormap(reduced_palette)

    # 在heatmap或其他需要色彩映射的图表中应用
    # sns.heatmap(data_matrix, cmap=cmap_reduced)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=0.3, cmap=cmap_reduced, annot_kws={"fontsize": 8})
    # sns.heatmap(data=data, annot=True, fmt="d", linewidths=0.3, cmap="RdBu_r", cbar_kws = {"orientation": "horizontal"})

    plt.title("SHAP Values Correlation Matrix")
    fig = plt.savefig(os.path.join(store_dir_path, "cool_warm.png"), dpi=300, format='png')
    plt.close(fig)
    # plt.show()

    # ========================== 创建热力图 去掉空 ==========================
    plt.figure(figsize=(10, 10))
    # full_palette = sns.color_palette("RdBu_r", 256)
    #
    # # 我们可以选择其中的一部分来创建一个新的ListedColormap
    # start_index = int(0.2 * len(full_palette))  # 开始位置（20%）
    # end_index = int(0.8 * len(full_palette))  # 结束位置（80%）
    # reduced_palette = full_palette[start_index:end_index]
    #
    # # 创建新的ListedColormap
    # cmap_reduced = mcolors.ListedColormap(reduced_palette)

    # 在heatmap或其他需要色彩映射的图表中应用
    # sns.heatmap(data_matrix, cmap=cmap_reduced)
    sns.heatmap(correlation_matrix_without_zero, annot=True, fmt=".2f", linewidths=0.3, cmap=cmap_reduced, annot_kws={"fontsize": 8})
    # sns.heatmap(data=data, annot=True, fmt="d", linewidths=0.3, cmap="RdBu_r", cbar_kws = {"orientation": "horizontal"})

    plt.title("SHAP Values Correlation Matrix")
    fig = plt.savefig(os.path.join(store_dir_path, "cool_warm_without_zero_importance.png"), dpi=300, format='png')
    plt.close(fig)


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
