"""
此文件用于定义常量模型

用于定义:
    1、路径配置信息
    2、限制约束条件
    3、各类算法中需要使用的变量
"""

class AlgorithmParams:
    """算法超参数配置"""
    # 遗传算法参数(GA)
    TRAY_POP_SIZE = 50                        # 种群大小
    TRAY_MUTATION_RATE = 0.25                  # 变异概率
    TRAY_ELITE_RATIO = 0.25                    # 精英保留比例
    MIN_PALLET_FILL = 0.8                      # 最小托盘面积填充率
    HYBRID_MAX_ITER = 1000                     # 最大迭代次数

    # 蚁群算法参数(ACO)
    ACO_ANTS_NUM = 150                          # 基准蚂蚁数量(根据问题规模动态调整)
    ACO_ANTS_DYNAMIC = lambda n: int(n**0.5)    # 动态蚂蚁数量公式
    ACO_EVAPORATION = 0.1                       # 信息素挥发系数(典型值0.1-0.5)
    ACO_PHEROMONE_DECAY = 1.0                   # 信息素强度(信息素初始强度，但需动态衰减)
    ACO_ALPHA = 1.0                             # 信息素重要程度(典型值1.0-2.0)
    ACO_BETA = 1.0                              # 启发式函数权重(典型值1.0-2.0)


    # 粒子群算法参数(PSO)
    PSO_INERTIA = (0.95, 0.3)           # 惯性权重动态范围（初始 → 终值）
    PSO_POP_SIZE = 30                   # 种群大小
    PSO_MAX_ITER = 500                  # 最大迭代次数
    PSO_COGNITVE_WEIGHT = 1.2           # 个体学习因子权重(典型值0.7-1.2)
    PSO_SOCIAL_WEIGHT = 1.2             # 社会学习因子权重(典型值0.7-1.2)
    PSO_MAX_VELOCITY = 1000             # 最大速度(典型值10-1000)

    # 模拟退火算法参数(SA)
    SA_INIT_TEMP = 500                  # 初始温度(需与目标函数量级匹配)
    SA_COOLING_RATE = 0.90              # 降温速率(典型值0.85-0.99)
    SA_MIN_TEMP = 1e-6                  # 最小温度(需与目标函数量级匹配)
    SA_TEMP_DECAY = "exponential"       # 降温方式（exponential/linear）
    SA_BOLTZMANN = 1e-3                 # Boltzmann 常数, 通常是 1e-5 到 1e-2 之间


    # 多目标遗传算法(NSGA-Ⅱ)
    NSGA_II_POP_SIZE = 50        # 种群大小
    NSGA_II_CROSSOVER_RATE = 0.7  # 交叉概率(交叉概率通常0.6-0.9)
    NSGA_II_MUTATION_RATE = 0.01  # 变异概率(变异概率通常0.01-0.1)
    NSGA_II_ELITES = 0.2          # 精英保留比例
    NSGA_II_MAX_GEN = 500         # 最大迭代次数

    # 混合算法全局控制
    HYBRID_MAX_ITER = 1500            # 最大迭代次数
    HYBRID_EARLY_STOP = 20            # 早停轮次


    SPECIAL_RULE_PENALTY_WEIGHT = 0.3  # 特殊规则惩罚权重

    # 约束惩罚权重
    CONSTRAINT_PENALTY = 0.5            # 违反约束的惩罚值


class PathConfig:
    """路径配置"""
    DATABASE_URI = "sqlite:///container_optimization.db"    # 数据库文件路径
    OUTPUT_DIR = "./output/"                                # 输出文件路径
    LOG_DIR = "./logs/"                                     # 日志文件路径

class BusinessRules:
    """业务规则"""
    FRAGILE_STACK_LIMIT = {
        0: 0,                                                                           # 非常易碎，禁止堆叠（0层）                                                  
        1: 2,                                                                           # 高易碎，允许堆叠2层
        2: 3,                                                                           # 较易碎，允许堆叠3层
        3: 5                                                                            # 不易碎，允许堆叠5层
    }                                                                                   # 易碎货物堆叠限制(堆叠层数)
    MAX_FRAGILITY = max(FRAGILE_STACK_LIMIT.keys())                                     # 最大易碎等级定义值，用于标准化计算（对应系统定义的最高易碎等级）
    MAX_WEIGHT = 10000                                                                  # 统允许的最大单件货物重量（kg），用于优先级计算的重量标准化
    MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER =  2100  # 普柜海运排托堆码高度限制
    MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER = 2500      # 高柜/超高柜海运排托堆码高度限制
    MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER = 2250       # 普柜海运散货堆码高度限制
    MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER = 2650          # 高柜/超高柜海运散货堆码高度限制
    MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_AIR_FREIGHT = 1500                          # 航空排拖堆码高度限制
    PALLET_LIMIT_LATERAL_OF_BULK_CARGO = {"min": 2310, "max": 2325}                     # 散货托盘最大宽度(横向宽度)
    PALLET_LIMIT_LONGITUDENAL = 1200                                                    # 托盘最大长度(纵向长度)
    PALLET_GAP = {"longitudinal": 20, "lateral": 100}                                   # 托盘横纵间距(纵向间距，横向间距)
    PALLET_GAP_CONTAINER = {"left": 0, "right": 0, "front": 0, "back": 50}              # 托盘与集装箱间距(左间距，右间距, 前间距, 后间距)
    GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET = 10                                            # 货物与托盘最小边缘间距
    SUPPORT_AREA_MIN_LIMIT = 0.95                                                       # 支撑面积最小限制


    SIZE_TOLERANCE = 5                                                                  # 尺寸容差校验容差（毫米），允许货物与装载空间存在±5mm的尺寸偏差
    SEA_OFFSET_LIMIT = 0.05                                                             # 海运重心偏移限制
    AIR_OFFSET_LIMIT = 0.03                                                             # 空运重心偏移限制
    PALLET_HEIGHT = 150                                                                 # 托盘高度（毫米）
