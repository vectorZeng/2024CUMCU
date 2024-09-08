import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体以支持中文标签
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义不同检测情况下的参数，包括次品率、检测成本、市场售价等
cases = [
    {'零配件1次品率': 0.1, '零配件2次品率': 0.1, '成品次品率': 0.1, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件2次品率': 0.2, '成品次品率': 0.2, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件2次品率': 0.1, '成品次品率': 0.1, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件2次品率': 0.2, '成品次品率': 0.2, '零配件1检测成本': 1, '零配件2检测成本': 1,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件2次品率': 0.2, '成品次品率': 0.1, '零配件1检测成本': 8, '零配件2检测成本': 1,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 5},
    {'零配件1次品率': 0.05, '零配件2次品率': 0.05, '成品次品率': 0.05, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '零配件1购买单价': 4, '零配件2购买单价': 18, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 40}
]


# 计算总利润
def total_profit_cal(case, n, BUY, detect_parts1, detect_parts2, detect_final, dismantle):
    """
    计算总成本和总利润
    参数:
    case (dict): 包含当前情况的各种参数。
    detect_parts1 (bool): 是否检测零配件1。
    detect_parts2 (bool): 是否检测零配件2。
    detect_final (bool): 是否检测成品。
    dismantle (bool): 是否拆解不合格成品。
    返回:
    tuple: 包含总成本和缺陷率。
    """
    n_parts1 = n  # 假设零配件1数量
    n_parts2 = n  # 假设零配件2数量
    # 零配件检测成本
    cost_parts1 = n_parts1 * (case['零配件1检测成本'] if detect_parts1 else 0 + case['零配件1购买单价']) if BUY else 0
    cost_parts2 = n_parts2 * (case['零配件2检测成本'] if detect_parts2 else 0 + case['零配件2购买单价']) if BUY else 0

    # 计算未检测零配件情况下的损失
    # loss_parts1 = (n_parts1 * case['零配件1次品率']) * (case['装配成本']) if not detect_parts1 else 0
    # loss_parts2 = (n_parts2 * case['零配件2次品率']) * (case['装配成本']) if not detect_parts2 else 0

    # 装配前的零件数
    before_final_n_parts1 = n_parts1 * ((1 - case['零配件1次品率']) if detect_parts1 else 1)
    before_final_n_parts2 = n_parts2 * ((1 - case['零配件2次品率']) if detect_parts2 else 1)

    # 成品数量
    n_final_products = min(before_final_n_parts1, before_final_n_parts2)

    # 计算装配成本
    cost_assembly = n_final_products * case['装配成本']

    # 计算检测成品成本
    cost_final = n_final_products * case['成品检测成本'] if detect_final else 0

    # 零件加工前的正品率
    standard_rate1 = 1 if detect_parts1 else (1 - case['零配件1次品率'])
    standard_rate2 = 1 if detect_parts2 else (1 - case['零配件2次品率'])

    # 更新后的成品次品率
    updated_defective_rate = 1 + (case['成品次品率'] - 1)*(standard_rate1 * standard_rate2)

    # 不合格的成品
    defective_final = n_final_products * updated_defective_rate

    # 计算不检测成品的调换损失
    loss_final = defective_final * case['调换损失'] if not detect_final else 0

    # 拆解成本
    dismantle_cost = defective_final * case['拆解费用'] if dismantle else 0

    # 拆解后能挽回的损失
    if dismantle and (defective_final > 2):
         a, b, dismantle_revenue = total_profit_cal(case, defective_final, False, detect_parts1, detect_parts2, detect_final, dismantle)
    else:
         dismantle_revenue = 0


    #卖出去的件数
    n_sell = n_final_products * (1 - updated_defective_rate)

    # 收入
    total_gain = case['市场售价'] * n_sell + dismantle_revenue  # (if detect_final else 1)

    # 计算总成本
    total_cost = (cost_parts1 + cost_parts2 + cost_assembly +
                  cost_final + loss_final + dismantle_cost)

    # 利润
    total_profit = round(total_gain - total_cost)



    return round(total_cost), round(total_gain), total_profit


# 定义所有检测策略的组合
strategies = []
for detect_parts1 in [True, False]:
    for detect_parts2 in [True, False]:
        for detect_final in [True, False]:
            for dismantle in [True, False]:
                # strategies;
                strategies.append({
                    "detect_parts1": detect_parts1,
                    "detect_parts2": detect_parts2,
                    "detect_final": detect_final,
                    "dismantle": dismantle
                })

# 生成策略说明
strategy_explanations = []
for i, strategy in enumerate(strategies):
    strategy_explanations.append(
        f"策略 {i + 1}: 检测零配件 1 {'是' if strategy['detect_parts1'] else '否'}，"
        f"检测零配件 2 {'是' if strategy['detect_parts2'] else '否'}，"
        f"检测成品 {'是' if strategy['detect_final'] else '否'}，"
        f"拆解不合格成品 {'是' if strategy['dismantle'] else '否'}"
    )

    # 存储每种情况的成本和缺陷率
costs = []
profits = []

# 遍历每种情况并计算成本和缺陷率
for i, case in enumerate(cases):
    print(f"\n情况 {i + 1}:")
    case_costs = []
    case_total_profit = []
    for j, strategy in enumerate(strategies):
        total_cost, total_gain, total_profit = total_profit_cal(case, 100, True, **strategy)
        case_costs.append(total_cost)
        case_total_profit.append(total_profit)
        print(f"{strategy_explanations[j]}: 总成本 = {total_cost:.1f}, 总收入 = {total_gain:.1f},总利润 = {total_profit:.1f}")
    costs.append(case_costs)
    profits.append(case_total_profit)

    # 将利润转换为NumPy数组以便于处理
costs = np.array(costs)
profits = np.array(profits)


# # 绘制不同情况和策略下的总成本比较图
# plt.figure(figsize=(10, 6))
# for i in range(costs.shape[1]):
#     plt.plot(range(1, costs.shape[0] + 1), costs[:, i], marker='o', label=f"策略 {i + 1} - 总成本")
# plt.xlabel("情况")
# plt.ylabel("总成本")
# plt.title("不同情况和策略下的总成本比较")
# plt.legend()
# plt.grid(True)
# plt.show()
#%%
plt.figure(figsize=(15, 9))
for i in range(profits.shape[1]):
    plt.plot(range(1, profits.shape[0] + 1), profits[:, i], marker='o', label=f"决策{i + 1}")
plt.xlabel("情况", fontsize=25)
plt.ylabel("总利润", fontsize=25)
# plt.title("不同情况和策略下的总利润比较")
# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', labelsize=20)  # 设置 x 和 y 轴刻度字体大小
plt.legend(loc='upper left', bbox_to_anchor=(0.99, 1), fontsize=15)
plt.grid(True)
# 保存图形为 .eps 文件
plt.savefig("pics/问题二：不同决策方案在不同情况下的总利润曲线.eps", format='eps')
plt.savefig("pics/问题二：不同决策方案在不同情况下的总利润曲线.png", format='png')
plt.show()

# 找到每一行的最大值及其索引
max_values = np.max(profits, axis=1)  # 每一行的最大值
max_indices = np.argmax(profits, axis=1)  # 每一行最大值的列索引

# 输出结果
for row in range(profits.shape[0]):
    col = max_indices[row]
    print(f"情况 {row + 1}: 使用决策方案 {col + 1}，所需总成本{costs[row,col]:.1f}元，可获得最大利润{max_values[row]:.1f}元")


#%%
    # 提取第一行数据
    first_row_data = profits[0, :]  # 取第一行的前6个策略数据

    # 策略标签
    strategies = [f"{i + 1}" for i in range(16)]

    # 绘制直方图
    plt.figure(figsize=(17, 8))

    bars = plt.bar(strategies, first_row_data, color='skyblue')

    # 在每个柱子上添加数据标签
    for bar in bars:
        yval = bar.get_height()  # 获取柱子的高度
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}',
                 ha='center', va='bottom', fontsize=15)  # 在柱子上方添加文本

    # 设置坐标轴标签和标题
    plt.xlabel("决策方案", fontsize=25)
    plt.ylabel("利润", fontsize=25)
    # plt.title("策略利润直方图", fontsize=20)

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', labelsize=15)

    # 显示网格
    plt.grid(axis='y')

    plt.savefig("pics/问题二：不同决策方案在情况1下的总利润直方图.eps", format='eps')
    plt.savefig("pics/问题二：不同决策方案在情况1下的总利润直方图.png", format='png')
    # 显示图形
    plt.show()

#%%
    # 提取第一行数据
    first_row_data = profits[2, :]  # 取第一行的前6个策略数据

    # 策略标签
    strategies = [f"{i + 1}" for i in range(16)]

    # 绘制直方图
    plt.figure(figsize=(17, 8))

    bars = plt.bar(strategies, first_row_data, color='skyblue')

    # 在每个柱子上添加数据标签
    for bar in bars:
        yval = bar.get_height()  # 获取柱子的高度
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}',
                 ha='center', va='bottom', fontsize=15)  # 在柱子上方添加文本

    # 设置坐标轴标签和标题
    plt.xlabel("决策方案", fontsize=25)
    plt.ylabel("利润", fontsize=25)
    # plt.title("策略利润直方图", fontsize=20)

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', labelsize=15)

    # 显示网格
    plt.grid(axis='y')

    plt.savefig("pics/问题二：不同决策方案在情况3下的总利润直方图.eps", format='eps')
    plt.savefig("pics/问题二：不同决策方案在情况3下的总利润直方图.png", format='png')
    # 显示图形
    plt.show()
