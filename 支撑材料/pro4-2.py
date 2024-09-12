
#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from matplotlib import rcParams

# 设置中文字体以便于图表显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 定义零配件的属性，包括次品率、购买单价和检测成本
components = {
    '零配件1': {'次品率': 0.1296, '购买单价': 2, '检测成本': 1},
    '零配件2': {'次品率': 0.0741, '购买单价': 8, '检测成本': 1},
    '零配件3': {'次品率': 0.0926, '购买单价': 12, '检测成本': 2},
    '零配件4': {'次品率': 0.1065, '购买单价': 2, '检测成本': 1},
    '零配件5': {'次品率': 0.0873, '购买单价': 8, '检测成本': 1},
    '零配件6': {'次品率': 0.0825, '购买单价': 12, '检测成本': 2},
    '零配件7': {'次品率': 0.1133, '购买单价': 8, '检测成本': 1},
    '零配件8': {'次品率': 0.1120, '购买单价': 12, '检测成本': 2},
}

# 定义半成品的属性，包括次品率、装配成本、检测成本和拆解费用
semi_products = {
    '半成品1': {'次品率': 0.0972, '装配成本': 8, '检测成本': 4, '拆解费用': 6},
    '半成品2': {'次品率': 0.1389, '装配成本': 8, '检测成本': 4, '拆解费用': 6},
    '半成品3': {'次品率': 0.1211, '装配成本': 8, '检测成本': 4, '拆解费用': 6},
}

# 定义成品的属性，包括次品率、装配成本、检测成本、拆解费用和市场售价
final_product = {
    '成品': {'次品率': 0.1736, '装配成本': 8, '检测成本': 6, '拆解费用': 10, '售价': 200, '调换损失': 40}
}

def total_profit_cal(n, BUY,  components, semi_products, final_product, detect_components,
                            detect_semi, semi_dismantle, detect_final, dismantle):
    """
    计算预期总成本和缺陷率。
    参数:
    components (dict): 零配件信息。
    semi_products (dict): 半成品信息。
    final_product (dict): 成品信息。
    detect_components (list): 是否检测各零配件的布尔列表。
    detect_semi (list): 是否检测各半成品的布尔列表。
    detect_final (bool): 是否检测成品的布尔值。
    dismantle (bool): 是否拆解不合格成品的布尔值。
    返回:
    tuple: 总成本和缺陷率。
    """
    n_comp = np.zeros(8)  # 变成半成品之前的零件个数
    n_semi = np.zeros(3)  # 变成成品之前的半成品个数
    n_semi_dismantle = np.zeros(3)
    # 零件加工前的正品率
    standard_rate = np.zeros(8)

    total_cost = 0  # 初始化总成本
    # total_defective_rate = 1  # 初始化总缺陷率

    # 计算零配件成本和总缺陷率
    for i, (comp_name, comp_data) in enumerate(components.items()):

        if i < 3:
            n_separate = n[0]  # 第一个零件组
        elif i < 6:
            n_separate = n[1]   # 第二个零件组
        else:
            n_separate = n[2]   # 第三个零件组
        cost = n_separate * comp_data['购买单价'] if BUY else 0  # 零配件的购买成本
        if detect_components[i]:  # 如果检测该零配件
            cost += n_separate * comp_data['检测成本']  # 加上检测成本
            standard_rate[i] = 1
        else:
            standard_rate[i] = 1 - comp_data['次品率']
        n_comp[i] = standard_rate[i] * n_separate

        total_cost += cost  # 累加到总成本

    updated_defective_semi_rate = np.zeros(3)  # 创建一个包含3个零的数组

    # 计算乘积并存储在 n_semi 中
    updated_defective_semi_rate[0] = semi_products['半成品1']['次品率']
    updated_defective_semi_rate[1] = semi_products['半成品2']['次品率']
    updated_defective_semi_rate[2] = semi_products['半成品3']['次品率']

    # 组装成的半成品个数
    n_semi[0] = np.min(n_comp[0:2])  # 第1-3个值的最小值
    n_semi[1] = np.min(n_comp[3:5])  # 第4-6个值的最小值
    n_semi[2] = np.min(n_comp[6:7])  # 第7-8个值的最小值

    defective_semi = np.zeros(3)

    # 半成品的正频率
    semi_standard_rate = np.zeros(3)

    # 计算半成品成本和总缺陷率
    for i, (semi_name, semi_data) in enumerate(semi_products.items()):
        cost = n_semi[i] * semi_data['装配成本']  # 半成品的装配成本

        if detect_semi[i]:  # 如果检测该半成品
            defective_semi[i] = n_semi[i] * updated_defective_semi_rate[i]
            cost += 4 * n_semi[i]  # 加上检测成本
            semi_standard_rate[i] = 1
        else:
            semi_standard_rate[i] = 1 - updated_defective_semi_rate[i]
        # n_semi_dismantle[i] = (1 - semi_standard_rate[i]) * n_semi[i]
        n_semi[i] = semi_standard_rate[i] * n_semi[i]
        total_cost += cost  # 累加到总成本

    if np.all(detect_semi) and semi_dismantle and (np.min(defective_semi) > 2):  # 如果所有半成品均已检测且需要拆解

        cost = np.sum(defective_semi) * 6  # 加上拆解费用

        # 拆解后，重新计算相关零件费用
        dismantled_comp_cost, dismantled_gain, dismantled_profit = total_profit_cal(
            defective_semi, False, components, semi_products, final_product,
            detect_components, detect_semi, semi_dismantle,
            detect_final, dismantle  # 为了避免重复拆解
        )
        total_cost += cost - dismantled_profit

    updated_defective_final_rate = final_product['成品']['次品率']

    # 最终的成品数量
    n_final = np.min(n_semi)
    # 卖出去的成品只有正品
    n_sell = n_final * (1 - updated_defective_final_rate)
    # 不合格的成品
    defective_final = n_final * updated_defective_final_rate

    # 计算成品的成本
    # final_defective_rate = final_product['成品']['次品率'] * (0.5 if detect_final else 1)  # 成品的次品率
    final_cost = n_final * final_product['成品']['装配成本']  # 成品的装配成本

    if detect_final:  # 如果检测成品
        final_cost += n_final * final_product['成品']['检测成本']  # 加上检测成本

    else:              # 如果不检测成品
        final_cost += defective_final * final_product['成品']['调换损失']

    total_cost += final_cost  # 累加到总成本

    if dismantle and (defective_final > 2):  # 如果拆解不合格成品
        total_cost += defective_final * final_product['成品']['拆解费用']  # 加上拆解费用
        # 拆解后，重新计算相关零件费用
        dismantled_final_cost, dismantled_final_gain, dismantled_final_profit = total_profit_cal(
            defective_final * np.ones(3), False, components, semi_products, final_product,
            detect_components, detect_semi, semi_dismantle,
            detect_final, dismantle  # 为了避免重复拆解
        )
        total_cost += - dismantled_final_profit
    # 收入
    total_gain = final_product['成品']['售价'] * n_sell  #* ((1 - updated_defective_rate) if detect_final else 1) + dismantle_revenue

    # 利润
    total_profit = round(total_gain - total_cost)

    # 返回总成本和利润
    return round(total_cost), round(total_gain), total_profit

    # 生成所有可能的检测组合


component_combinations = list(product([True, False], repeat=8))  # 零配件检测组合
semi_combinations = list(product([True, False], repeat=3))  # 半成品检测组合
semi_dismantle_combinations = list(product([True, False], repeat=1))  # 半成品拆解组合
final_combinations = list(product([True, False], repeat=2))  # 成品检测和拆解组合

strategy_descriptions = []  # 策略描述列表
results = []  # 结果列表


def generate_strategy_description(detect_components, detect_semi, semi_dismantle, detect_final, dismantle):
    """
    生成策略描述。

    参数:
    detect_components (list): 零配件检测布尔列表。
    detect_semi (list): 半成品检测布尔列表。
    detect_final (bool): 成品检测布尔值。
    dismantle (bool): 拆解布尔值。

    返回:
    str: 策略描述字符串。
    """
    description = ""
    for i, detect in enumerate(detect_components):
        description += f"检测零配件 {i + 1}，" if detect else f"不检测零配件 {i + 1}，"
    for i, detect in enumerate(detect_semi):
        description += f"检测半成品 {i + 1}，" if detect else f"不检测半成品 {i + 1}，"
    for i, dismantle in enumerate(semi_dismantle):
        description += f"拆解半成品，" if semi_dismantle else f"不拆解半成品，"

    description += "检测成品，" if detect_final else "不检测成品，"
    description += "拆解不合格成品" if dismantle else "不拆解不合格成品"
    return description


# 策略编号初始化
strategy_number = 1
# 遍历所有组合，计算成本和缺陷率
for comp_comb in component_combinations:
    for semi_comb in semi_combinations:
        for final_comb in final_combinations:
            for semi_dismantle_comb in semi_dismantle_combinations:
                semi_dismantle = semi_dismantle_comb  # 当前半成品拆解组合
                detect_components = comp_comb  # 当前零配件检测组合
                detect_semi = semi_comb  # 当前半成品检测组合
                detect_final = final_comb[0]  # 当前成品检测状态
                dismantle = final_comb[1]  # 当前拆解状态

                # 计算当前策略的总成本和利润
                total_cost, total_gain, total_profit = total_profit_cal([100, 100, 100], True, components, semi_products, final_product, detect_components,
                                                               detect_semi, semi_dismantle, detect_final, dismantle)

                # 生成当前策略的描述
                strategy_description = generate_strategy_description(detect_components, detect_semi, semi_dismantle, detect_final,
                                                                     dismantle)
                results.append([strategy_number, strategy_description, total_cost, total_gain, total_profit])  # 将结果添加到列表
                strategy_number += 1  # 策略编号递增

# 将结果转换为DataFrame并输出
df = pd.DataFrame(results, columns=['决策方案编号', '决策方案描述', '总成本', '总收入', '总利润'])
print(df.head())  # 打印前五条策略结果

# 将结果保存到Excel文件
df.to_excel('问题四决策方案结果.xlsx', index=False)
print("所有策略已经保存到 '决策方案结果.xlsx' 文件中。")


#%%
# 绘制总成本比较图
plt.figure(figsize=(10, 6))
plt.plot(df['决策方案编号'], df['总成本'], marker='o', linestyle='-', color='blue', label="总成本")
plt.xlabel("决策方案编号", fontsize=25)
plt.ylabel("总成本", fontsize=25)
# plt.title("不同决策方案下的总成本比较", fontsize=25)
plt.grid(True)
plt.legend(fontsize=20)
plt.savefig("pics/问题四：不同决策方案的总成本折线图.eps", format='eps')
plt.savefig("pics/问题四：不同决策方案的总成本折线图.png", format='png')
plt.show()

# 绘制总利润比较图
plt.figure(figsize=(10, 6))
plt.plot(df['决策方案编号'], df['总利润'], marker='x', linestyle='--', color='red', label="总利润")
plt.xlabel("决策方案编号", fontsize=25)
plt.ylabel("总利润", fontsize=25)
# plt.title("不同决策方案下的总利润比较")
plt.grid(True)
plt.legend(fontsize=20)
plt.savefig("pics/问题四：不同决策方案的总利润折线图.eps", format='eps')
plt.savefig("pics/问题四：不同决策方案的总利润折线图.png", format='png')
plt.show()

#%%


# 找出总利润最大的10行
top_10_profit = df.nlargest(10, '总利润')

# 输出决策方案描述及其 '总成本', '总收入', '总利润'
#print(top_10_profit[['决策方案描述']].to_string(index=False), "所得的总利润为：", top_10_profit[['总利润']].to_string(index=False))

# 绘制直方图
plt.figure(figsize=(16, 10))
bars = plt.bar(range(len(top_10_profit)), top_10_profit['总利润'], color='skyblue')

# 在每个条形上标出数据
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=25)

# 设置图表标题和标签
#plt.title('总利润最大的10个决策方案')
plt.xlabel('决策方案编号', fontsize=25)
plt.ylabel('总利润', fontsize=25)

# 使用字符标注横坐标
plt.xticks(range(len(top_10_profit)), top_10_profit['决策方案编号'].astype(str), rotation=45)

# plt.tight_layout()  # 自动调整布局
# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', labelsize=20)
plt.grid(axis='y')
plt.savefig("pics/问题四：总利润最大的10个决策方案直方图.eps", format='eps')
plt.savefig("pics/问题四：总利润最大的10个决策方案直方图.png", format='png')
# 显示图表
plt.show()
top_10_profit.to_excel('问题四：决策方案中利润最高的10个.xlsx', index=False)

#%%
# 找出总利润最大的10行
top_10_profit = df.nlargest(10, '总利润')

# 输出决策方案描述及其 '总成本', '总收入', '总利润'
#print(top_10_profit[['决策方案描述']].to_string(index=False), "所得的总利润为：", top_10_profit[['总利润']].to_string(index=False))

# 绘制直方图
plt.figure(figsize=(16, 10))
bars = plt.bar(range(len(top_10_profit)), top_10_profit['总成本'], color='skyblue')

# 在每个条形上标出数据
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=25)

# 设置图表标题和标签
#plt.title('总成本最大的10个决策方案')
plt.xlabel('决策方案编号', fontsize=25)
plt.ylabel('总成本', fontsize=25)

# 使用字符标注横坐标
plt.xticks(range(len(top_10_profit)), top_10_profit['决策方案编号'].astype(str), rotation=45)

# plt.tight_layout()  # 自动调整布局
# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', labelsize=20)
plt.grid(axis='y')
plt.savefig("pics/问题四：总成本最大的10个决策方案直方图.eps", format='eps')
plt.savefig("pics/问题四：总成本最大的10个决策方案直方图.png", format='png')
# 显示图表
plt.show()
top_10_profit.to_excel('问题四：决策方案中成本最高的10个.xlsx', index=False)


