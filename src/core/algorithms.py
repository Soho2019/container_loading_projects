import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
import math
from concurrent.futures import ProcessPoolExecutor
from pulp import LpMaximize, LpProblem, LpVariable
from typing import Union, Tuple, List, Dict
from scipy.spatial import KDTree, Voronoi
from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

from config.constants import AlgorithmParams, BusinessRules
from core.domain import ContainerSpec, ProductsSpec, PalletSpec, LoadingPoint, Solution, Particle, Placement
from core.constraints import ConstraintChecker
from database.converters import decode_rotations


# ----------------------------------- 三大指标衡量函数 -----------------------------------
def calculate_volume_utilization(products: List[ProductsSpec], contaniner: ContainerSpec, use_pallet: bool) -> float:
    """计算货箱利用率(拆分排拖情况和不排拖情况)"""
    total_volume = sum(product.volume for product in products)
    if not use_pallet:
        # 不排拖情况，直接计算货物体积占比
        return total_volume / contaniner.volume
    else:
        # 排拖情况，需扣除托盘高度（托盘高度统一为150mm）
        effective_container_height = contaniner.height - BusinessRules.PALLET_HEIGHT
        if effective_container_height <= 0:
            return 0.0

        # 利用率 = (货物体积 / 集装箱总体积(扣除托盘高度))
        effective_container_volume = contaniner.length * contaniner.width * effective_container_height
        return total_volume / effective_container_volume

def calculate_weight_utilization(products: List[ProductsSpec], container: ContainerSpec) -> float:
    """计算载重利用率"""
    total_weight = sum(product.weight for product in products)
    if total_weight > container.max_weight:
        raise ValueError(f'载重超限: 总重量 {total_weight} 超过容器最大载重 {container.max_weight}')
    return total_weight / container.max_weight

def calculate_center_offset(products: List[ProductsSpec], container: ContainerSpec, positions: list, return_total: bool = False) -> Union[float, Tuple[float, float, float]]:
    ### 已修改错误：每件货物的重心时，X、Y、Z轴直接与货物的length、width、height绑定，如果货物摆放进行了旋转，会导致结果计算错误
    # position: 货物左前下角坐标列表和实际摆放后的长宽高, 每个元素为 (x, y, z, rotated_dimensions)
    # return_total：是否返回总偏移量，默认为 False
    """计算重心偏移量（动态传入坐标）"""
    total_weight = sum(product.weight for product in products)
    if total_weight <= 0:
        return  (float('inf'), float('inf'), float('inf'))  # 避免除零错误
    
    weighted_x, weighted_y, weighted_z = 0.0, 0.0, 0.0

    for product, pos in zip(products, positions):
        # 提取旋转后的实际尺寸
        x, y, z, (l, w, h) = pos                # 要求 positions 传入旋转后的真实尺寸
        # 计算货物的几何中心坐标
        center_x = l / 2 + x            # 使用实际长度l计算X方向中心
        center_y = w / 2 + y
        center_z = h / 2 + z

        # 累加加权坐标
        weighted_x += center_x * product.weight
        weighted_y += center_y * product.weight
        weighted_z += center_z * product.weight
    
    # 计算整体重心
    center_gx = weighted_x / total_weight
    center_gy = weighted_y / total_weight
    center_gz = weighted_z / total_weight

    # 安全范围约束
    x_min, x_max = container.length * 0.45, container.length * 0.55
    y_min, y_max = container.width * 0.45, container.width * 0.55
    z_min, z_max = container.height * 0.45, container.height * 0.55

    # 计算各轴偏移量
    offset_x = max(0, x_min - center_gx, center_gx - x_max)
    offset_y = max(0, y_min - center_gy, center_gy - y_max)
    offset_z = max(0, z_min - center_gz, center_gz - z_max)

    return (offset_x + offset_y + offset_z) if return_total else (offset_x, offset_y, offset_z)

# ------------------------------- 帕累托前沿处理 -------------------------------
class ParetoFront:
    """简易帕累托前沿管理"""
    def __init__(self, solutions: List[Solution]):
        self.solutions = solutions
    
    def top(self, n: int) -> List[Solution]:
        return sorted(self.solutions, key=lambda x: x.fitness, reverse=True)[:n]

# ------------------------------- 支配关系判断 -------------------------------
def dominates(a: Solution, b: Solution) -> bool:
    """判断解a是否支配解b"""
    better_volume = a.volume_utilization >= b.volume_utilization
    better_weight = a.weight_utilization >= b.weight_utilization
    better_stability = a.stability_score >= b.stability_score

    return (better_volume and better_weight and better_stability and 
            (a.volume_utilization > b.volume_utilization or
             a.weight_utilization > b.weight_utilization or
             a.stability_score > b.stability_score))



# ----------------------------------- 托盘选择，基于遗传算法(双层优化+动态剪枝) -----------------------------------
@dataclass
class PalletSolution:
    """托盘布局方案"""
    pallets: List[PalletSpec]
    positions: List[Tuple[float, float, float]]             # 托盘在集装箱内的坐标(x, y, z)
    utilization: float = 0.0                                # 托盘利用率
    stability_score: float = 0.0                            # 稳定性评分
    fitness: float = 0.0                                    # 综合适应度

    def __post_init__(self):
        """在初始化后计算适应度"""
        self.fitness = 0.6 * self.utilization + 0.4 * self.stability_score

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'pallets': [p.to_dict() for p in self.pallets],
            'positions': self.positions,
            'utilization': self.utilization,
            'stability_score': self.stability_score,
            'fitness': self.fitness
        }

class GeneticAlgorithmOptimizer:
    """遗传算法优化器(替代原PallentOptimizerGA)"""
    def __init__(
            self, 
            container: ContainerSpec,
            products: List[ProductsSpec],
            candidate_pallets: List[PalletSpec],
            params: dict = None):
        
        self.container = container
        self.products = products
        self.candidate_pallets = candidate_pallets
        
        # 遗传算法参数
        self.pop_size = AlgorithmParams.TRAY_POP_SIZE
        self.max_gen = AlgorithmParams.HYBRID_MAX_ITER
        self.elite_ratio = AlgorithmParams.TRAY_ELITE_RATIO
        self.mutation_rate = AlgorithmParams.TRAY_MUTATION_RATE
        
        # 预处理货物数据(按体积降序)
        self.sorted_products = sorted(products, key=lambda x: -x.volume)
    
    def optimize(self) -> Solution:
        """主优化流程"""
        population = self._init_population()
        
        for _ in range(self.max_gen):
            # 评估适应度
            evaluated = [(ind, self._fitness(ind)) for ind in population]
            evaluated.sort(key=lambda x: x[1], reverse=True)
            
            # 精英保留
            elites = [ind for ind, _ in evaluated[:int(self.elite_ratio * self.pop_size)]]
            
            # 生成新一代
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(evaluated)
                p2 = self._tournament_select(evaluated)
                child = self._crossover(p1, p2)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_pop.append(child)
            
            population = new_pop
        
        best = max(population, key=lambda x: self._fitness(x))
        pallet_solution = self._decode_solution(best)

        # 计算重量利用率
        total_weight = sum(p.weight for p in self.products)
        weight_utilization = total_weight / self.container.max_weight if self.container.max_weight > 0 else 0.0

        return Solution(
            items=pallet_solution.pallets,
            positions=pallet_solution.positions,
            volume_utilization=pallet_solution.utilization,
            weight_utilization=weight_utilization,
            stability_score=pallet_solution.stability_score,
            fitness=pallet_solution.fitness
        )
    
    def _init_population(self) -> List[List[PalletSpec]]:
        """初始化种群"""
        return [self._generate_individual() for _ in range(self.pop_size)]
    
    def _generate_individual(self) -> List[PalletSpec]:
        """生成随机个体"""
        solution = []
        used_length = 0
        max_length = self.container.length - self.container.door_reserve
        
        while used_length < max_length:
            valid_pallets = [
                p for p in self.candidate_pallets 
                if p.length <= (max_length - used_length)
            ]
            if not valid_pallets:
                break
                
            pallet = random.choice(valid_pallets)
            solution.append(pallet)
            used_length += pallet.length + BusinessRules.PALLET_GAP['longitudinal']
        
        return solution
    
    def _fitness(self, individual: List[PalletSpec]) -> float:
        """计算适应度"""
        if not individual:
            return 0.0
            
        # 面积利用率
        total_area = sum(p.length * p.width for p in individual)
        container_area = self.container.length * self.container.width
        area_ratio = total_area / container_area
        
        # 填充率
        fill_ratio = self._calc_fill_ratio(individual)
        
        return 0.6 * area_ratio + 0.4 * fill_ratio
    
    def _calc_fill_ratio(self, pallets: List[PalletSpec]) -> float:
        """计算填充率"""
        total_filled = 0
        for pallet in pallets:
            eff_area = (pallet.length - 2 * BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET) * (pallet.width - 2 * BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET)
            
            filled = sum(p.volume for p in self.products if 
                        p.length <= pallet.length and 
                        p.width <= pallet.width)
            
            total_filled += min(filled, eff_area)
        
        return total_filled / sum(p.length*p.width for p in pallets)
    
    def _decode_solution(self, individual: List[PalletSpec]) -> PalletSolution:
        """解码为托盘布局方案"""
        positions = []
        x, y = 0, 0
        max_row_height = 0
        
        for pallet in individual:
            if x + pallet.length > self.container.length - self.container.door_reserve:
                y += max_row_height + BusinessRules.PALLET_GAP['lateral']
                x = 0
                max_row_height = 0
            
            positions.append((x, y, 0))  # z坐标固定为0
            x += pallet.length + BusinessRules.PALLET_GAP['longitudinal']
            max_row_height = max(max_row_height, pallet.width)
        
        # 计算利用率
        total_area = sum(p.length * p.width for p in individual)
        container_area = self.container.length * self.container.width
        utilization = total_area / container_area if container_area > 0 else 0.0
        
        return PalletSolution(
            pallets=individual,
            positions=positions,
            utilization=utilization,
            stability_score=self._calculate_stability(individual, positions)
        )
    
    def _calculate_stability(self, pallets: List[PalletSpec], positions: List[Tuple[float, float, float]]) -> float:
        """计算稳定性得分"""
        if not pallets:
            return 0.0
        
        total_weight = sum(p.max_weight for p in pallets)
        weighted_x = sum((pos[0] + p.length / 2) * p.max_weight for p, pos in zip(pallets, positions))
        weighted_y = sum((pos[1] + p.width / 2) * p.max_weight for p, pos in zip(pallets, positions))
        
        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight
        
        # 计算偏移量
        offset_x = abs(center_x - self.container.length / 2)
        offset_y = abs(center_y - self.container.width / 2)
        
        # 标准化为0-1分数
        max_offset = max(self.container.length, self.container.width) / 2
        return 1 - (offset_x + offset_y) / (2 * max_offset)
    
    def _tournament_select(self, evaluated: List, k=3) -> List[PalletSpec]:
        candidates = random.sample(evaluated, k)
        return max(candidates, key=lambda x: x[1])[0]
    
    def _crossover(self, p1: List[PalletSpec], p2: List[PalletSpec]) -> List[PalletSpec]:
        if len(p1) == 0 or len(p2) == 0:
            return p1 if len(p1) > 0 else p2        # 返回非空父代
        
        max_cut = min(len(p1), len(p2))
        if max_cut <= 1:
            return p1                               # 直接返回父代避免无效切割
        
        cut = random.randint(1, max_cut - 1)
        return p1[:cut] + p2[cut:]
    
    def _mutate(self, individual: List[PalletSpec]) -> List[PalletSpec]:
        if len(individual) == 0:
            return individual                               # 直接返回空个体，避免操作
        
        idx = random.randint(0, len(individual) - 1)
        valid_pallets = [p for p in self.candidate_pallets if p.length <= individual[idx].length * 1.2]
        if not valid_pallets:
            return individual
        new_pallet = random.choice(valid_pallets)
        return individual[:idx] + [new_pallet] + individual[idx+1:]

# ----------------------------------- 塔装载启发式算法 -----------------------------------
class TowerPackingAlgorithm:
    """塔装载启发式算法(含货运种类区分接口)"""
    def __init__(
            self, 
            container: ContainerSpec, 
            products: List[ProductsSpec], 
            pallet: PalletSpec, 
            transport_type: str = 'sea',        # sea/air, 海运/空运区分标志
            cargo_type: str = 'bulk',           # pallet/bulk, 托盘/散货区分标志
            container_type: str = 'general',    # general/high, 普柜/高柜、超高柜区分标志
    ):
        """
        参数：
        container: 集装箱规格（单位：毫米）
        products: 待装载货物列表（单位：毫米）
        pallet: 托盘规格（单位：毫米）
        transport_type: 运输类型 sea/air
        cargo_type: 货物类型 pallet/bulk
        container_type: 集装箱类型 general/high
        """
        self.container = container
        self.products = products
        self.pallet = pallet
        self.transport_type = transport_type
        self.cargo_type = cargo_type
        self.container_type = container_type
        self.max_height = container.height - pallet.height if pallet.height else 0

        # 初始化业务规则参数
        self._init_business_rules(transport_type, cargo_type, container_type)

        # 状态记录
        self.fragile_stack = defaultdict(int)
        self.positions = []
        self.current_height = 0         # 使用示例变量跟踪高度
    
    def _init_business_rules(self, transport_type: str, cargo_type: str, container_type: str):
        """根据运输场景初始化业务规则"""
        # 高度限制规则选择
        if transport_type == 'sea':
            if cargo_type == 'pallet':
                self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER
                if container_type == 'high':
                    self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER
            else:
                self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER
                if container_type == 'high':
                    self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER
        elif transport_type == 'air':
            self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_AIR_FREIGHT
        else:
            raise ValueError('运输类型无效')
        
        # 间隙限制规则选择
        self.edge_gap = BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET
        if cargo_type == 'bulk':
            self.pallet_gap = BusinessRules.PALLET_LIMIT_LATERAL_OF_BULK_CARGO
        
    def optimize(self) -> dict:
        """主优化流程"""
        # 按优先级排序(易碎 > 重量 > 高度)
        sorted_products = sorted(
            self.products,
            # key=self._calculate_priority,                                                 # 暂时注释掉
            key=lambda p: (p.length * p.width * p.height, p.weight),
            reverse=True
        )

        for product in sorted_products:
            # 获取允许的摆放姿态
            valid_orientations = self._get_valid_orientations(product)
            # 选择高度最小的姿态以增加堆叠层数
            valid_orientations.sort(key=lambda dim: dim[2])
            for dim in valid_orientations:
                if self._can_place_product(product, dim):
                    self._place_product(product, dim)
                    break
        return self._build_result()
    
    def _get_valid_orientations(self, product: ProductsSpec) -> List[Tuple]:
        """生成有效货物姿态(优化性能)"""
        return [
            (l, w, h)
            for (l, w, h) in product.allowed_rotations
            if self._is_valid_dimension(l, w, h)
        ]

    def _is_valid_dimension(self, l: int, w: int, h: int) -> bool:
        """检查尺寸是否符合业务规则（增加容差处理）"""
        eff_length = self.pallet.length - 2 * self.edge_gap + BusinessRules.SIZE_TOLERANCE
        eff_width = self.pallet.width - 2 * self.edge_gap + BusinessRules.SIZE_TOLERANCE
        return l <= eff_length and w <= eff_width

    def _can_place_product(self, product: ProductsSpec, dim: tuple) -> bool:
        """检查货物是否可以放置"""
        # 高度检查
        if self.current_height + dim[2] > self.max_height:
            return False
        # 易碎品堆叠限制
        if product.fragility in BusinessRules.FRAGILE_STACK_LIMIT:
            allowed_layers = BusinessRules.FRAGILE_STACK_LIMIT[product.fragility]

            # 特别处理非常易碎货物
            if product.fragility == 0:
                if self.fragile_stack[product.id] >= 0:
                    return False
            else:
                if self.fragile_stack[product.id] >= allowed_layers:
                    return False
        if not self.positions:
            return True
        # 支撑面积检查(使用重叠面积而非比例之和)
        if self.positions:
            last = self.positions[-1]
            overlap_area = min(dim[0], last['dimensions'][0]) * min(dim[1], last['dimensions'][1])
            min_area = product.length * product.width * BusinessRules.SUPPORT_AREA_MIN_LIMIT
            return overlap_area >= min_area * 0.5                                               # 临时修改 * 0.5
        return True
    
    def _place_product(self, product: ProductsSpec, dim: tuple):
        """执行放置操作(标准化数据结构)"""
        if len(dim) == 2:
            dim = (dim[0], dim[1], self.pallet.height)

        self.positions.append({
            'product': product,
            # 'sku': product.sku,
            'position': (self.edge_gap, self.edge_gap, self.current_height),
            'dimensions': dim,
            # 'fragility': product.fragility,
        })
        self.current_height += dim[2]
        if product.fragility > 0:
            self.fragile_stack[product.id] += 1

    def _calculate_priority(self, product: ProductsSpec) -> float:
        """计算货物优先级(易碎性 > 重量 > 高度)"""
        fragility = product.fragility / BusinessRules.MAX_FRAGILITY
        weight = 1 - (product.weight / max(p.weight for p in self.products))
        height = 1 - (product.height / self.max_height)
        return (fragility * 0.6) + (weight * 0.3) + (height * 0.1)
    
    def _build_result(self) -> dict:
        """构建结果（添加元数据）"""
        return {
            'transport_type': self.transport_type,
            'positions': [
                {
                    'product': entry['product'],  # 确保包含product对象
                    'dimensions': entry['dimensions'],
                    'position': entry['position']
                } 
                for entry in self.positions
            ],
            'utilization': self._calculate_utilization(),
            'stability': self._check_stability(),
            'fragile_counts': dict(self.fragile_stack),
            'total_height': self.current_height
        }
    
    def _calculate_utilization(self) -> float:
        """体积利用率计算（增加防零除保护）"""
        used = sum(item['dimensions'][0] * item['dimensions'][1] * item['dimensions'][2] for item in self.positions)
        available = (self.pallet.length - 2 * self.edge_gap) * (self.pallet.width - 2 * self.edge_gap) * self.max_height
        return used / available if available > 0 else 0
    
    def _check_stability(self) -> dict:                                                                                                 # 存疑
        """稳定性检查(包含海运/空运特殊规则)"""
        total_weight = sum(p['product'].weight for p in self.positions)
        if total_weight == 0:
            return {'stable': False, 'reason': 'no_products'}
        
        # 计算重心坐标
        weighted_x = sum((p['position'][0] + p['dimensions'][0]/2) * p['product'].weight for p in self.positions)
        weighted_y = sum((p['position'][1] + p['dimensions'][1]/2) * p['product'].weight for p in self.positions)
        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight
        
        # 动态阈值
        if self.transport_type == 'sea':
            x_limit = self.container.length * BusinessRules.SEA_OFFSET_LIMIT
            y_limit = self.container.width * BusinessRules.SEA_OFFSET_LIMIT
        else:
            x_limit = self.container.length * BusinessRules.AIR_OFFSET_LIMIT
            y_limit = self.container.width * BusinessRules.AIR_OFFSET_LIMIT
        
        return {
            'stable': abs(center_x) < x_limit and abs(center_y) < y_limit,
            'actual_offset': (center_x, center_y),
            'allowed_offset': (x_limit, y_limit),
            'threshold': BusinessRules.SEA_OFFSET_LIMIT if self.transport_type == 'sea' else BusinessRules.AIR_OFFSET_LIMIT
        }

# ----------------------------------- 二维装载点优化算法 -----------------------------------
class BinPacking2D:
    """二维装载优化器(Voronoi空隙分析 + DBSCAN聚类)"""                                                          # 待修改
    def __init__(self, container_size: Tuple[float, float], global_offset: Tuple[float, float] = (0, 0)):
        """
        参数：
        container_width: 装载区域宽度（单位：毫米）
        container_height: 装载区域高度（单位：毫米）
        global_offset: 全局全局坐标偏移量（单位：毫米, 用于结果转换）
        """
        self.width, self.height = container_size
        self.global_offset = global_offset

        # 初始化装载点(考虑间隙规则)
        self.gap_left = BusinessRules.PALLET_GAP_CONTAINER['left']
        self.gap_right = BusinessRules.PALLET_GAP_CONTAINER['right']
        self.gap_front = BusinessRules.PALLET_GAP_CONTAINER['front']
        self.gap_back = BusinessRules.PALLET_GAP_CONTAINER['back']
        self.gap_lateral = BusinessRules.PALLET_GAP['lateral']
        self.gap_longitudinal = BusinessRules.PALLET_GAP['longitudinal']

        self.points = []
        self._init_points()

    def _init_points(self):
        """初始化装载点(考虑间隙规则)"""
        effective_width = self.width - self.gap_left - self.gap_right
        effective_height = self.height - self.gap_front - self.gap_back

        self.points.append({
            'x': self.gap_left,
            'y': self.gap_front,
            'width': effective_width,
            'height': effective_height,
            'active': True
        })
    
    def hybrid_optimize(self, items:List[dict]) -> dict:
        """
        混合优化流程
        items: 待装载货物列表，每个元素包含：
        'dimensions': (长， 宽， 高)
        'fragility': 易碎等级
        """
        # 阶段1：基础贪婪装载
        base_solution = self.greedy_packing(items)

        # 阶段2：智能优化（示例使用SA）
        optimized = self.simulated_annealing(base_solution)

        # 转换全局坐标
        return self._convert_to_global(optimized)
    
    def greedy_packing(self, items: List[dict]) -> List[dict]:
        """贪婪装载算法"""
        sorted_items = sorted(items, key=lambda x: -(x['dimensions'][0] * x['dimensions'][1]))

        result = []

        for item in sorted_items:
            placed = False
            for point in sorted(self.active_points, key=lambda p: -p['width'] * p['height']):
                if self._can_place(item, point):
                    self._place_item(item, point)
                    result.append(self._create_placement(item, point))
                    placed = True
                    break
            if not placed:
                print(f"警告：无法放置货物{item.get('id', '未知ID')}")
        return result
    
    def simulated_annealing(self, initial_solution):
        """模拟退火算法优化框架"""
        current = initial_solution
        current_energy = self._energy(current)

        for temp in self._cooling_schedule():
            neighbor = self._generate_neighbor(current)
            neighbor_energy = self._energy(neighbor)

            if self._accept_solution(current_energy, neighbor_energy, temp):
                current = neighbor
                current_energy = neighbor_energy
        return current
    
    def _energy(self, solution) -> float:
        """评估方案质量: 负空间利用率 + 稳定性惩罚"""
        utilization = self._calculate_utilization(solution)
        # stability = self._calculate_stability(solution)
        # return -(utilization * 0.8 + stability * 0.2)                # 负利用率转为最小化问题
        return -utilization
    
    def  _generate_neighbor(self, solution):
        """生成邻域解（交换两个随机物品位置）"""
        new_sol = solution.copy()                                                                                                           # 存疑
        if len(new_sol) >= 2:
            i, j = np.random.choice(len(new_sol), 2, replace=False)
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        return new_sol
    
    def _convert_to_global(self, solution) -> dict:
        """转换局部坐标到全局坐标系"""
        total_volume = sum(item['dimensions'][0] * item['dimensions'][1] * item['dimensions'][2] for item in solution)

        return {
            'positions': [{
                'global_x': self.global_offset[0] + pos['x'],
                'global_y': self.global_offset[1] + pos['y'],
                'dimensions': pos['dimensions'],
                'product_id': pos['product_id'],
            } for pos in solution],
            'utilization': self._calculate_utilization(solution),
            'total_volume': total_volume
        }
    
    @property
    def active_points(self):
        return [p for p in self.points if p['active']]
    
    def _can_place(self, item, point) -> bool:
        """检查是否可以放置"""
        item_w = item['dimensions'][0] + self.gap_lateral
        item_h = item['dimensions'][1]+ self.gap_longitudinal
        return (
            item_w <= point['width'] and
            item_h <= point['height'] and
            point['active']
        )
    
    def _place_item(self, item, point):
        """执行放置操作并分割空间"""
        point['active'] = False

        # 生成右侧剩余区域
        remaining_width = point['width'] - (item['dimensions'][0] + self.gap_lateral)
        if remaining_width > 0:
            self.points.append({
                'x': point['x'] + item['dimensions'][0] + self.gap_lateral,
                'y': point['y'],
                'width': remaining_width,
                'height': point['height'],
                'active': True
            })
        
        # 生成后方剩余区域
        remaining_height = point['height'] - (item['dimensions'][1] + self.gap_longitudinal)
        if remaining_height > 0:
            self.points.append({
                'x': point['x'],
                'y': point['y'] + item['dimensions'][1] + self.gap_longitudinal,
                'width': point['width'],
                'height': remaining_height,
                'active': True
            })
        
    def _create_placement(self, item, point) -> dict:                                                                                                    # 存疑  
        """创建放置记录"""
        # 如果item直接包含id字段
        product_id = item.get('id') 
        # 如果id在product对象内
        if not product_id and 'product' in item:
            product_id = item['product'].id
        return {
            'product_id': product_id,
            'x': point['x'],
            'y': point['y'],
            'dimensions': item['dimensions'],
            'fragility': item.get('fragility', 0)
        }
    
    def _calculate_utilization(self, solution) -> float:
        """计算空间利用率"""
        used = sum(item['dimensions'][0] * item['dimensions'][1] for item in solution)
        # 计算有效区域（扣除间隙）
        effective_width = self.width - BusinessRules.PALLET_GAP_CONTAINER['left'] - BusinessRules.PALLET_GAP_CONTAINER['right']
        effective_height = self.height - BusinessRules.PALLET_GAP_CONTAINER['front'] - BusinessRules.PALLET_GAP_CONTAINER['back']
        available = effective_width * effective_height
        return used / available if available > 0 else 0
    
    def _cooling_schedule(self):
        """模拟退火降温曲线"""
        return np.linspace(1.0, 0.01, num=100)
    
    def _accept_solution(self, current_e, new_e, temp) -> bool:
        """接受准则"""
        if new_e < current_e:
            return True
        return np.exp((current_e - new_e) / temp) > np.random.random()

# ----------------------------------- 分层优化控制器 -----------------------------------
class HybridOptimizer:
    """协同优化控制器(GA+PSO+SA混合优化, 支持遗传算法+塔装载+二维优化的分层协调)"""
    def __init__(self, container: ContainerSpec, products: List[ProductsSpec], candidate_pallets: List[PalletSpec]):
        """
        参数：
        container: 集装箱规格（单位：毫米）
        products: 待装载货物列表（单位：毫米）
        candidate_pallets: 可选托盘类型列表（单位：毫米）
        """
        self.container = container
        self.products = products
        self.candidate_pallets = candidate_pallets
        self.use_pallet = len(candidate_pallets) > 0

        # 优化状态记录
        self.best_solution = None
        self.optimization_history = []

    def optimize(self) -> Dict:
        """执行混合优化流程（GA → PSO → SA）"""
        # 阶段1：遗传算法生成初始解
        ga_solutions = self._run_ga()
        self._log_stage("GA", ga_solutions)

        # 阶段2：PSO优化
        pso_solution = self._run_pso(ga_solutions)
        self._log_stage("PSO", [pso_solution])

        # 阶段3：SA优化
        sa_solution = self._run_sa(pso_solution)
        self._log_stage("SA", [sa_solution])

        # 验证约束并返回结果
        if self._validate_constraints(sa_solution):
            return self._build_final_result(sa_solution)
        else:
            raise ValueError("最终方案违反约束条件")
    
    def _run_ga(self) -> List[Solution]:
        """遗传算法生成初始解"""
        ga = GeneticAlgorithmOptimizer(
            container=self.container,
            products=self.products,
            candidate_pallets=self.candidate_pallets
        )
        pallet_solutions = [ga.optimize() for _ in range(AlgorithmParams.TRAY_POP_SIZE)]

        return pallet_solutions

    def _run_pso(self, initial_solutions: List[Solution]) -> Solution:
        """PSO优化局部解"""
        pso = PSO(container=self.container, products=self.products)
        return pso.optimize(initial_solutions, self.container)

    def _run_sa(self, initial_solution: Solution) -> Solution:
        """SA优化全局解"""
        sa = SA(container=self.container, products=self.products)
        return sa.optimize(initial_solution, self.container)

    def _validate_constraints(self, solution: Solution) -> bool:
        """验证所有业务约束"""
        checker = ConstraintChecker(self.container, self.use_pallet)
        
        # 分离货物和托盘
        if self.use_pallet:
            pallets = [p for p in solution.items if isinstance(p, PalletSpec)]
            products = [p for p in solution.items if isinstance(p, ProductsSpec)]
            positions = solution.positions
            pallet_positions = positions[:len(pallets)]
            product_positions = positions[len(pallets):]
            
            # 检查托盘专用约束
            if not self._check_pallet_gaps(pallets, pallet_positions):
                return False
            if not self._check_fragile_stack(products, product_positions):
                return False
            
            return checker.check_all(products, pallets, product_positions, pallet_positions)
        else:
            return checker.check_all(solution.items, [], solution.positions, [])

    def _check_pallet_gaps(self, pallets: List[PalletSpec], positions: List[tuple]) -> bool:
        """检查托盘间隙约束"""
        for pallet, (x, y, _) in zip(pallets, positions):
            # 检查与集装箱边界的间隙
            if (x < BusinessRules.PALLET_GAP_CONTAINER['left'] or 
                y < BusinessRules.PALLET_GAP_CONTAINER['front'] or
                self.container.length - (x + pallet.length) < BusinessRules.PALLET_GAP_CONTAINER['back'] or
                self.container.width - (y + pallet.width) < BusinessRules.PALLET_GAP_CONTAINER['right']):
                return False
        return True

    def _check_fragile_stack(self, products: List[ProductsSpec], positions: List[tuple]) -> bool:
        """检查易碎品堆叠约束"""
        stack_map = defaultdict(int)
        for product, (_, _, z) in zip(products, positions):
            if product.fragility in BusinessRules.FRAGILE_STACK_LIMIT:
                max_layers = BusinessRules.FRAGILE_STACK_LIMIT[product.fragility]
                if stack_map[product.id] >= max_layers:
                    return False
                stack_map[product.id] += 1
        return True

    def _build_final_result(self, solution: Solution) -> Dict:
        """构建与原格式兼容的输出"""
        # 分离托盘和货物
        pallets = []
        product_positions = []
        for item, pos in zip(solution.items, solution.positions):
            if isinstance(item, PalletSpec):
                pallets.append(item)
            else:
                product_positions.append({
                    'product': item,
                    'position': pos[:3],  # (x,y,z)
                    'dimensions': pos[3]  # (l,w,h)
                })

        # 计算利用率
        total_volume = sum(p.volume for p in solution.items)
        utilization = total_volume / self.container.volume

        return {
            'pallets': pallets,
            'positions': product_positions,
            'utilization': utilization,
            'stability': solution.stability_score
        }

    def _log_stage(self, stage: str, solutions: List[Solution]):
        """记录优化过程"""
        for sol in solutions:
            self.optimization_history.append({
                'stage': stage,
                'fitness': sol.fitness,
                'volume_utilization': sol.volume_utilization,
                'weight_utilization': sol.weight_utilization,
                'stability_score': sol.stability_score
            })   

# ----------------------------------- ACO + PSO + SA  -----------------------------------
class ACO:
    """蚁群优化算法，生成初始解"""
    def __init__(self):
        self.num_ants = AlgorithmParams.ACO_ANTS_NUM
        self.pheromone = defaultdict(float)
        self.decay = AlgorithmParams.ACO_PHEROMONE_DECAY
        self.alpha = AlgorithmParams.ACO_ALPHA
        self.beta = AlgorithmParams.ACO_BETA
    
    def generate_solutions(self, container: ContainerSpec, products: List[ProductsSpec]) ->List[Solution]:
        solutions = []
        for _ in range(self.num_ants):
            solution = self._construct_solution(container, products)
            solutions.append(solution)
            self._update_pheromone(solution)
        return solutions
    
    def _construct_solution(self, container: ContainerSpec, products: List[ProductsSpec]) -> Solution:
        remaining_volume = container.volume
        placements = []
        remaining_products = products.copy()

        # 按启发式信息排序（体积/重量比）
        remaining_products.sort(key=lambda x: -(x.volume / x.weight) if x.weight > 0 else -x.volume)

        current_position = [0, 0, 0]                 # 当前位置坐标
        current_layer_height = 0

        while remaining_products and remaining_volume > 0:
            # 选择下一个货物
            next_product = self._select_next_product(remaining_products, current_position)
            if not next_product:
                break

            # 选择最佳旋转方向
            best_rotation = self._select_best_rotation(next_product, current_position, container)

            # 放置货物
            placement = Placement(
                product=next_product,
                position=tuple(current_position),
                dimensions=best_rotation
            )
            placements.append(placement)

            # 更新状态
            remaining_volume -= next_product.volume
            remaining_products.remove(next_product)

            # 更新当前位置
            current_position[0] += best_rotation[0]                     # 沿长度方向移动
            if current_position[0] + best_rotation[0] > container.length:
                current_position[0] = 0
                current_position[1] += best_rotation[1]                 # 沿宽度方向移动

                if current_position[1] + best_rotation[1] > container.width:
                    current_position[1] = 0
                    current_position[2] += current_layer_height         # 沿高度方向移动
                    current_layer_height = 0
            current_layer_height = max(current_layer_height, best_rotation[2])
        
        items = [p.product for p in placements]
        positions = [(p.position[0], p.position[1], p.position[2], p.dimensions) for p in placements]
        
        return Solution(items=items, positions=positions)
    
    def _select_next_product(self, products: List[ProductsSpec], current_pos: List[int]) -> ProductsSpec:
        """概率选择下一个货物"""
        if not products:
            return None
        
        # 计算启发式信息（体积优先）
        heuristic = [p.volume for p in products]

        # 计算信息素
        pheromone = [self.pheromone.get(p.id, 1.0) for p in products]

        # 计算选择概率
        probabilities = [(ph ** self.alpha) * (h ** self.beta) for ph, h in zip(pheromone, heuristic)]
        total  = sum(probabilities)
        if total <= 0:
            return random.choice(products)
        
        probabilities = [p / total for p in probabilities]

        # 轮盘赌选择
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return products[i]
        return products[-1]
    
    def _select_best_rotation(self, product: ProductsSpec, pos: List[int], container: ContainerSpec) -> tuple:
        """计算最佳旋转方向"""
        valid_rotations = []
        for rot in product.allowed_rotations:
            # 检查是否超出集装箱边界
            if (pos[0] + rot[0] <= container.length and
                pos[1] + rot[1] <= container.width and
                pos[2] + rot[2] <= container.height):
                valid_rotations.append(rot)
        
        if not valid_rotations:
            return product.allowed_rotations[0]         # 默认返回第一个旋转方向
        
        # 选择最节省空间的旋转方向（最小高度优先）
        return min(valid_rotations, key=lambda x: x[2])
    
    def _update_pheromone(self, solution: Solution):
        """更新信息素"""
        # 挥发旧信息素
        for k in list(self.pheromone.keys()):
            self.pheromone[k] *= (1.0 - self.decay)
        
        # 增加新信息素（基于解决方案质量）
        if solution.placements:
            quality = solution.volume_utilization
            for placement in solution.placements:
                product_id = placement.product.id
                self.pheromone[product_id] += quality

class PSO:
    """粒子群优化算法，优化局部解"""
    def __init__(self, container: ContainerSpec, products: List[ProductsSpec]):
        self.container = container
        self.products = products
        self.pop_size = AlgorithmParams.PSO_POP_SIZE
        self.inertia = AlgorithmParams.PSO_INERTIA[0]               # 使用初始惯性权重
        self.cognitive_wight = AlgorithmParams.PSO_COGNITVE_WEIGHT
        self.social_weight = AlgorithmParams.PSO_SOCIAL_WEIGHT
        self.max_velocity = AlgorithmParams.PSO_MAX_VELOCITY
        self.population = []
        self.global_best = None         # 添加全局最优解记录

    def optimize(self, initial_solutions: List[Solution], container: ContainerSpec) -> Solution:
        # 初始化种群
        self._initialize_population(initial_solutions, container)

        for iteration in range(AlgorithmParams.PSO_MAX_ITER):
            # 动态调整惯性权重
            self._adjust_inertia(iteration)

            for particle in self.population:
                # 更新速度和位置
                self._update_velocity(particle)
                self._update_position(particle, container)

                # 评估新位置
                self._evaluate(particle, container)

                # 更新个体最优
                if particle.current_fitness > particle.best_fitness:
                    particle.best_position = deepcopy(particle.position)
                    particle.best_fitness = particle.current_fitness
                    particle.best_solution = deepcopy(particle.solution)
                
                # 更新全局最优
                if self.global_best is None or particle.best_fitness > self.global_best.best_fitness:
                    self.global_best = deepcopy(particle)
        
        return self.global_best.best_solution
    
    def _initialize_population(self, initial_solutions: List[Solution], container: ContainerSpec):
        """初始化粒子群"""
        self.population = []

        # 确保初始解有效
        valid_solutions = [sol for sol in initial_solutions if sol and sol.items]
        if not valid_solutions:
            valid_solutions = [Solution(items=[], positions=[])]
        
        # 创建粒子
        for i in range(self.pop_size):
            base_solution = valid_solutions[i % len(valid_solutions)]
            particle = Particle()

            # 编码解决方案（位置 = 货物顺序 + 旋转）
            particle.position = self._encode_solution(base_solution)
            particle.velocity = [random.uniform(-1, 1) for _ in range(len(particle.position))]

            particle.best_position = deepcopy(particle.position)
            particle.solution = deepcopy(base_solution)
            particle.best_solution = deepcopy(base_solution)

            # 初始评估
            self._evaluate(particle, container)
            particle.best_fitness = particle.current_fitness

            self.population.append(particle)
        
        # 初始化全局最优
        if self.population:
            self.global_best = deepcopy(max(self.population, key=lambda x: x.best_fitness))
        else:
            self.global_best = Particle()  # 提供一个默认粒子
    
    def _encode_solution(self, solution: Solution) -> list:
        """将解决方案编码为位置向量"""
        encoded = []
        for placement in solution.placements:
            # 编码：货物ID + 旋转索引 + 位置坐标
            encoded.extend([
                float(placement.product.id),
                float(placement.product.allowed_rotations.index(placement.dimensions)),
                placement.position[0],
                placement.position[1],
                placement.position[2]
            ])
        return encoded
    
    def _decode_solution(self, position: list, products: List[ProductsSpec]) -> Solution:
        """从位置向量解码解决方案"""
        placements = []
        product_dict = {p.id: p for p in products}

        i = 0
        while i < len(position):
            product_id = int(position[i])
            rot_idx = int(position[i + 1]) % len(product_dict[product_id].allowed_rotations)
            x, y, z = position[i + 2], position[i + 3], position[i + 4]

            placement = Placement(
                product=product_dict[product_id],
                position=(x, y, z),
                dimensions=product_dict[product_id].allowed_rotations[rot_idx]
            )
            placements.append(placement)
            i += 5
        
        return Solution(placements=placements)
    
    def _adjust_inertia(self, iteration: int):
        """动态调整惯性权重"""
        start, end = AlgorithmParams.PSO_INERTIA
        self.inertia = start - (start - end) * (iteration / AlgorithmParams.PSO_MAX_ITER)
    
    def _update_velocity(self, particle: Particle):
        """更新粒子速度"""
        for i in range(len(particle.velocity)):
            # 计算认知和社会部分
            r1, r2 = random.random(), random.random()
            cognitive = self.cognitive_wight * r1 * (particle.best_position[i] - particle.position[i])
            social = self.social_weight * r2 * (self.global_best.best_position[i] - particle.position[i])

            # 更新速度
            particle.velocity[i] = self.inertia * particle.velocity[i] + cognitive + social

            # 限制速度
            if abs(particle.velocity[i]) > self.max_velocity:
                particle.velocity[i] = self.max_velocity * (1 if particle.velocity[i] > 0 else - 1)
    
    def _update_position(self, particle: Particle, container: ContainerSpec):
        """更新粒子位置"""
        for i in range(len(particle.position)):
            particle.position[i] += particle.velocity[i]

        # 解码并修复位置
        try:
            particle.solution = self._decode_solution(particle.position, list({p.product for p in self.global_best.best_solution.placements}))
            particle.solution = self._repair_solution(particle.solution, container)
            particle.position = self._encode_solution(particle.solution)
        except (KeyError, IndexError):
            # 如果解码失败，恢复到之前的最佳位置
            particle.position = deepcopy(particle.best_position)
            particle.solution = deepcopy(particle.best_solution)
    
    def _repair_solution(self, solution: Solution, container: ContainerSpec) -> Solution:
        """修复无效的解决方案"""
        if not solution.placements:
            return solution
        
        # 移除超出集装箱的货物
        valid_placements = []
        used_products = set()
        total_volume = 0

        for placement in solution.placements:
            # 检查是否重复装载同一货物
            if placement.product.id in used_products:                                                                   # 同一货物装载多次如何处理？用id是否不妥？
                continue

            # 检查是否超出集装箱边界
            if (placement.position[0] + placement.dimensions[0] <= container.length and 
                placement.position[1] + placement.dimensions[1] <= container.width and
                placement.position[2] + placement.dimensions[2] <= container.height):
                valid_placements.append(placement)
                used_products.add(placement.product.id)
                total_volume += placement.product.volume

                # 检查是否超载
                if total_volume > container.volume * 1.1:               # 允许 10% 超额
                    break
        
        return Solution(placements=valid_placements)

    def _evaluate(self, particle: Particle, container: ContainerSpec):
        """评估粒子适应度"""
        if not particle.solution.items:
            particle.current_fitness = 0.0
            return
        
        # 计算体积利用率
        total_volume = sum(p.volume for p in particle.solution.items)
        volume_utilization = total_volume / container.volume if container.volume > 0 else 0.0

        # 计算重量利用率
        total_weight = sum(p.weight for p in particle.solution.items)
        weight_utilization = total_weight / container.max_weight if container.max_weight > 0 else 0.0

        # 计算稳定性
        stability = self._calculate_stability(particle.solution, container)

        # 综合适应度
        particle.current_fitness = (
            0.6 * volume_utilization +
            0.3 * weight_utilization +
            0.1 * stability
        )
        
    def _calculate_stability(self, solution: Solution, container: ContainerSpec) -> float:
        """计算稳定得分"""
        if not solution.placements:
            return 0.0
        
        total_weight = sum(p.product.weight for p in solution.placements)
        if total_weight <= 0:
            return 0.0
        
        # 计算重心
        center_x = sum(p.position[0] + p.dimensions[0] / 2 * p.product.weight for p in solution.placements) / total_weight
        center_y = sum(p.position[1] + p.dimensions[1] / 2 * p.product.weight for p in solution.placements) / total_weight

        # 计算偏移量（距离集装箱中心的距离）
        offset_x = abs(center_x - container.length / 2) / (container.length / 2)
        offset_y = abs(center_y - container.width / 2) / (container.width / 2)

        # 稳定性得分（偏移越小得分越高）
        return 1.0 - (offset_x + offset_y) / 2
        
class SA:
    """模拟退火算法，优化全局解"""
    def __init__(self, container: ContainerSpec, products: List[ProductsSpec]):
        self.container = container
        self.products = products
        self.temp = AlgorithmParams.SA_INIT_TEMP
        self.cooling_rate = AlgorithmParams.SA_COOLING_RATE
        self.min_temp = AlgorithmParams.SA_MIN_TEMP
        self.k = AlgorithmParams.SA_BOLTZMANN

    def optimize(self, initial_solution: Solution, container: ContainerSpec) -> Solution:
        current = deepcopy(initial_solution)
        current_energy = self._energy(current, container)
        best_solution = deepcopy(current)
        best_energy = current_energy

        while self.temp > self.min_temp:
            # 生成邻域解
            neighbor = self._perturb(current, container)
            neighbor_energy = self._energy(neighbor, container)

            # 决定是否接受新解
            if self._accept(current_energy, neighbor_energy):
                current = neighbor
                current_energy = neighbor_energy

                # 更新最佳解
                if neighbor_energy < best_energy:
                    best_solution = deepcopy(neighbor)
                    best_energy = neighbor_energy
            
            # 降温
            self.temp *= self.cooling_rate

        return best_solution

    def _perturb(self, solution: Solution, container: ContainerSpec) -> Solution:
        """扰动当前解生成邻域解"""
        new_solution = deepcopy(solution)

        if not new_solution.placements:
            return new_solution

        # 随机选择扰动方式
        mutation_type = random.choice([
            'swap',
            'rotate',
            'move',
            'add',
            'remove'
        ])

        if mutation_type == 'swap' and len(new_solution.placements) >= 2:
            # 交换两个货物的位置
            i, j = random.sample(range(len(new_solution.placements)), 2)
            new_solution.placements[i], new_solution.placements[j] = (new_solution.placements[j], new_solution.placements[i])
        elif mutation_type == 'rotate' and new_solution.placements:
            # 随机改变一个货物的旋转方向
            idx = random.randint(0, len(new_solution.placements) - 1)
            placement = new_solution.placements[idx]
            new_rotation = random.choice([
                rot for rot in placement.product.allowed_rotations
                if self._is_valid_placement(
                    placement.position,
                    rot,
                    container
                )
            ])
            new_solution.placements[idx].dimensions = new_rotation
        elif mutation_type == 'move' and new_solution.placements:
            # 移动一个货物的位置
            idx = random.randint(0, len(new_solution.placements) - 1)
            placement = new_solution.placements[idx]

            # 尝试新位置
            new_x = placement.position[0] + random.randint(-100, 100)
            new_y = placement.position[1] + random.randint(-100, 100)
            new_z = placement.position[2] + random.randint(-50, 50)

            # 确保新位置有效
            new_x = max(0, min(new_x, container.length - placement.dimensions[0]))
            new_y = max(0, min(new_y, container.width - placement.dimensions[1]))
            new_z = max(0, min(new_z, container.height - placement.dimensions[2]))

            new_solution.placements[idx].position = (new_x, new_y, new_z)
        
        elif mutation_type == 'add' and len(new_solution.placements) < len(solution.placements) * 1.5:
            # 尝试添加一个新货物（从当前未装载的货物中）
            loaded_ids = {p.product.id for p in new_solution.placements}
            available_products = [
                p for p in container.products if p.id not in loaded_ids
            ]

            if available_products:
                product = random.choice(available_products)
                rotation = random.choice(product.allowed_rotations)

                # 寻找有效位置
                for _ in range(10):                     # 最多尝试 10 次
                    x = random.uniform(0, container.length - rotation[0])
                    y = random.uniform(0, container.width - rotation[1])
                    z = random.uniform(0, container.height - rotation[2])

                    if self._is_valid_position(new_solution.placements, (x, y, z), rotation):
                        new_solution.placements.append(Placement(
                            product=product,
                            position=(x, y, z),
                            dimensions=rotation
                        ))
                        break
        
        elif mutation_type == 'romove' and len(new_solution.placements) > 1:
            # 随机移除一个货物
            idx = random.randint(0, len(new_solution.placements) - 1)
            new_solution.placements.pop(idx)
        
        return new_solution
    
    def _is_valid_placement(self, position: tuple, dimensions: tuple, container: ContainerSpec) -> bool:
        """检查放置是否有效"""
        return (
            position[0] + dimensions[0] <= container.length and
            position[1] + dimensions[1] <= container.width and
            position[2] + dimensions[2] <= container.height
        )
    
    def _is_valid_position(self, placements: List[Placement], position: tuple, dimensions: tuple) -> bool:
        """检查新位置是否与其他货物重叠"""
        new_min = position
        new_max = (
            position[0] + dimensions[0],
            position[1] + dimensions[1],
            position[2] + dimensions[2]
        )

        for placement in placements:
            existing_min = placement.position
            existing_max = (
                placement.position[0] + placement.dimensions[0],
                placement.position[1] + placement.dimensions[1],
                placement.position[2] + placement.dimensions[2]
            )

            # 检查是否有重叠
            if not (
                new_max[0] <= existing_min[0] or
                new_min[0] >= existing_max[0] or
                new_max[1] <= existing_min[1] or
                new_min[1] >= existing_max[1] or
                new_max[2] <= existing_min[2] or
                new_min[2] >= existing_max[2]
            ):
                return False
            
        return True
    
    def _energy(self, solution: Solution, container: ContainerSpec) -> float:
        """计算解决方案的能量（越低越好）"""
        if not solution.placements:
            return float('inf')
        
        # 计算体积利用率（反转，因为我们要最小化能量）
        total_volume = sum(p.product.volume for p in solution.placements)
        volume_utilization = total_volume / container.volume

        # 计算重量利用率
        total_weight = sum(p.product.weight for p in solution.placements)
        weight_utilization = total_weight / container.max_weight if container.max_weight > 0 else 0.0

        # 计算稳定性
        stability = self._calculate_stability(solution, container)

        # 能量函数（考虑未利用空间和稳定性）
        unused_volume = 1.0 - volume_utilization
        instability = 1.0 - stability

        return (
            0.7 * unused_volume +
            0.2 * instability +
            0.1 * (1.0 - weight_utilization)
        )
    
    def _calculate_stability(self, solution: Solution, container: ContainerSpec) -> float:
        """计算稳定性得分（与PSO中的相同）"""
        if not solution.placements:
            return 0.0
        
        total_weight = sum(p.product.weight for p in solution.placements)
        if total_weight <= 0:
            return 0.0
        
        # 计算重心
        center_x = sum((p.position[0] + p.dimensions[0] / 2) * p.product.weight for p in solution.placements) / total_weight
        center_y = sum((p.position[1] + p.dimensions[1] / 2) * p.product.weight for p in solution.placements) / total_weight

        # 计算偏移量（距离容器中心的距离）
        offset_x = abs(center_x - container.length / 2) / (container.length / 2)
        offset_y = abs(center_y - container.width / 2) / (container.width / 2)

        # 稳定性得分（偏移越小得分越高）
        return 1.0 - (offset_x + offset_y) / 2
    
    def _accept(self, current_energy: float, new_energy: float) -> bool:
        """决定是否接受新解"""
        if new_energy < current_energy:
            return True
        return math.exp((current_energy - new_energy) / (self.k * self.temp)) > random.random()

# ------------------------------- 花垛算法 -------------------------------
class FlowerStackOptimizer:
    """集成力学模型的花垛算法"""
    def __init__(self, container, products):
        self.container = container
        self.products = sorted(products, key=lambda x: x.fragility, reverse=True)                                                                  # 存疑
        self.layers = []
        self.current_z = 0

    def optimize(self):
        while self.products:
            current_layer = self._build_layer()
            self._adjust_centroid(current_layer)
            self.layers.append(current_layer)
        return self._generate_solution()
    
    def _generate_solution(self):
        """生成解决方案"""
        placements = []
        for layer in self.layers:
            for placement in layer:
                placements.append(placement)
        
        return {
            'placements': placements,
            'layers': self.layers,
        }
    
    def _build_layer(self):                                                                                                     # 存疑
        layer = []
        remaining_width = self.container.width
        current_x = 0

        while remaining_width > 0 and self.products:
            # 线性规划选择最优组合
            selected = self._select_by_lp(remaining_width)
            if not selected:
                break

            for item in selected:
                pos = (current_x, self.current_z)
                dimensions = (item.width, item.length, item.height)  
                layer.append(Placement(item, pos, dimensions, pallet_id=0))
                current_x += item.width + BusinessRules.PALLET_GAP['lateral']
                remaining_width -= item.width
                self.products.remove(item)
        return layer
    
    def _select_by_lp(self, max_width):                                                                                                     # 存疑
        """线性规划选择最优填充组合"""
        prob = LpProblem("GapFilling", LpMaximize)
        vars = [LpVariable(f"x{i}", cat='Binary') for i in range(len(self.products))]
        
        # 目标函数：最大化重量稳定性
        prob += sum(vars[i] * (self.products[i].weight * self.products[i].width) for i in range(len(vars)))
        
        # 约束条件
        prob += sum(vars[i] * self.products[i].width for i in range(len(vars))) <= max_width
        
        prob.solve()
        return [self.products[i] for i in range(len(vars)) if vars[i].value() == 1]

    def _adjust_centroid(self, layer):
        """确保质心投影在下层支撑区域内"""
        if not self.layers:
            return
            
        total_weight = sum(item.weight for item in layer)
        centroid_x = sum(item.x * item.weight for item in layer) / total_weight
        prev_layer_span = (self.layers[-1][0].x, self.layers[-1][-1].x + self.layers[-1][-1].width)
        
        # 调整位置使质心在支撑区域内
        if centroid_x < prev_layer_span[0]:
            delta = prev_layer_span[0] - centroid_x
            for item in layer:
                item.x += delta
        elif centroid_x > prev_layer_span[1]:
            delta = centroid_x - prev_layer_span[1]
            for item in layer:
                item.x -= delta

# ------------------------------- 特殊规则处理器 -------------------------------
class SpecialRuleEnforcer:                                                                                                  # 存疑
    """软约束处理：多品类托盘惩罚项"""
    def __init__(self, container, products):
        self.container = container
        self.products = products
        self.required_categories = set(p.category for p in products)

    def calculate_penalty(self, solution):
        # 检查是否存在包含所有品类的托盘
        category_count = defaultdict(set)
        for placement in solution.placements:
            category_count[placement.pallet_id].add(placement.product.category)
        
        penalty = 0
        for pallet_id, categories in category_count.items():
            if self.required_categories.issubset(categories):
                return 0
        # 惩罚项 = 缺失品类数 * 权重
        missing = len(self.required_categories - categories)
        return missing * AlgorithmParams.SPECIAL_RULE_PENALTY_WEIGHT

# ------------------------------- NSGA-II -------------------------------
class NSGAII:
    """多目标优化框架"""
    def __init__(self):
        self.pop_size = AlgorithmParams.NSGA_II_POP_SIZE
        self.crossover_rate = AlgorithmParams.NSGA_II_CROSSOVER_RATE
        self.mutation_rate = AlgorithmParams.NSGA_II_MUTATION_RATE
        self.elites = AlgorithmParams.NSGA_II_ELITES

    def optimize(self, population: List[Solution]) -> List[Solution]:
        for _ in range(AlgorithmParams.NSGA_II_MAX_GEN):
            # 非支配排序
            fronts = self._fast_non_domainated_sort(population)
            # 计算拥挤距离
            for front in fronts:
                self._crowding_distance_assignment(front)
            # 选择新一代
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) > self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: -getattr(x, 'crowding_distance', 0))
                    new_pop.extend(front[:self.pop_size - len(new_pop)])
                    break
            # 生成子代
            offspring = self._create_offspring(new_pop)
            population = new_pop + offspring
        return population
    
    def _fast_non_domainated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """非支配排序"""
        fronts = [[]]
        for ind in population:
            # 使用 setattr 动态添加属性
            setattr(ind, 'domination_count', 0)
            setattr(ind, 'dominated_set', [])

            for other in population:
                if dominates(ind, other):
                    ind.dominated_set.append(other)
                elif dominates(other, ind):
                    ind.domination_count += 1
            if ind.domination_count == 0:
                fronts[0].append(ind)

        i = 0
        while fronts[i]:
            next_front = []
            for ind in fronts[i]:
                for domained in ind.dominated_set:
                    domained.domination_count -= 1
                    if domained.domination_count == 0:
                        next_front.append(domained)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]
    
    def _crowding_distance_assignment(self, front: List[Solution]):
        """计算拥挤距离"""
        for ind in front:
            if not hasattr(ind, 'crowding_distance'):
                setattr(ind, 'crowding_distance', 0.0)
        
        for m in ['volume_utilization', 'weight_utilization', 'stability_score']:
            front.sort(key=lambda x: getattr(x, m))
            setattr(front[0], 'crowding_distance', float('inf'))
            setattr(front[-1], 'crowding_distance', float('inf'))

            for i in range(1, len(front) - 1):
                current_dist = getattr(front[i], 'crowding_distance', 0.0)
                delta = (getattr(front[i+1], m) - getattr(front[i-1], m))
                setattr(front[i], 'crowding_distance', current_dist + delta)
    
    def _create_offspring(self, population: List[Solution]) -> List[Solution]:
        """生成子代"""
        offspring = []
        while len(offspring) < len(population) * (1 - self.elites):
            # 选择父代
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
                # 变异
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                offspring.append(child)
        return offspring
    
    def _tournament_select(self, population: List[Solution]) -> Solution:
        """锦标赛选择"""
        candidates = random.sample(population, 2)
        return max(candidates, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """交叉操作"""
        child = deepcopy(parent1)
        if len(child.items) > 1 and len(parent2.items) > 1:
            # 随机交换部分物品
            idx = random.randint(1, min(len(child.items), len(parent2.items)) - 1)
            child.items[idx:] = parent2.items[idx:]
            child.positions[idx:] = parent2.positions[idx:]
        return child
    
    def _mutate(self, solution: Solution) -> Solution:
        """变异操作"""
        mutated = deepcopy(solution)
        if len(mutated.items) > 1:
            i, j = random.sample(range(len(mutated.items)), 2)
            mutated.items[i], mutated.items[j] = mutated.items[j], mutated.items[i]
            mutated.positions[i], mutated.positions[j] = mutated.positions[j], mutated.positions[i]
        return mutated


# ------------------------------- 并行评估器 -------------------------------
class ParallelEvaluator:
    """多进程并行评估器"""
    def __init__(self, container: ContainerSpec, num_workers=None):
        self.container = container
        self.pool = ProcessPoolExecutor(max_workers=num_workers)

    def evaluate(self, solutions: List[Solution]) -> List[Solution]:
        """并行评估解决方案"""
        futures = [self.pool.submit(self._evaluate_single, sol) for sol in solutions]
        return [f.result() for f in futures]
    
    def _evaluate_single(self, solution: Solution) -> Solution:
        """评估单个解决方案"""
        # 计算体积利用率
        total_volume = sum(p.volume for p in solution.items)
        solution.volume_utilization = total_volume / self.container.volume

        # 计算重量利用率
        total_weight = sum(p.weight for p in solution.items)
        solution.weight_utilization = total_weight / self.container.max_weight

        # 计算稳定性
        solution.stability_score = self._calculate_stability(solution)

        # 综合适应度
        solution.fitness = (
            0.5 * solution.volume_utilization +
            0.3 * solution.weight_utilization +
            0.2 * solution.stability_score
        )
        return solution
    
    def _calculate_stability(self, solution: Solution) -> float:
        """计算稳定性得分"""
        if not solution.items:
            return 0.0
        
        total_weight = sum(p.weight for p in solution.items)
        weighted_x = sum((pos[0] + dim[0] / 2) * p.weight for p, pos, dim in zip(solution.items, solution.positions, [p.allowed_rotations[0] for p in solution.items]))
        weighted_y = sum((pos[1] + dim[1] / 2) * p.weight for p, pos, dim in zip(solution.items, solution.positions, [p.allowed_rotations[0] for p in solution.items]))
        
        offset_x = weighted_x / total_weight
        offset_y = weighted_y / total_weight
                     
        # 计算偏移量
        offset_x = abs(offset_x - self.container.length / 2)
        offset_y = abs(offset_y - self.container.width / 2)

        # 标准化为0-1分数
        max_offset = max(self.container.length, self.container.width) / 2
        return 1 - (offset_x + offset_y) / (2 * max_offset)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):  
        """上下文管理器退出时关闭线程池"""
        self.pool.shutdown(wait=True)