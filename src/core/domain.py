"""
此文件定义与算法交互的领域模型

如集装箱类、货物类、托盘类等

长度单位为毫米(mm), 重量单位为千克(kg)
"""

from __future__ import annotations          # 支持类型注解的前向引用
from copy import deepcopy
from dataclasses import dataclass, field
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

@dataclass
class ContainerSpec:
    """集装箱规格"""
    id: int
    name: str
    length: float
    width: float
    height: float
    max_weight: float
    door_reserve: float = 50    # 门预留空间

    @property
    def volume(self) -> float:
        """计算体积(mm³)"""
        return self.length * self.width * self.height    # 体积

@dataclass
class ProductsSpec:
    """货物规格"""
    id: int
    sku: str
    frgn_name: str
    item_name: str
    length: float
    width: float
    height: float
    weight: float
    fragility: int                                      # 易碎等级
    allowed_rotations: List[Tuple[int, int, int]]       # 允许的旋转姿态
    category: str                                       # 货物类别

    @property
    def volume(self) -> float:
        """计算体积(mm³)"""
        return self.length * self.width * self.height    # 体积

@dataclass
class PalletSpec:
    """托盘规格"""
    id: int
    length: float
    width: float
    height: float
    max_weight: float

    @property
    def to_dict(self) -> Dict:
        return{
            'id': self.id,
            'length': self.length,
            'width': self.width,
            'height': self.height,
            'max_weight': self.max_weight
        }

@dataclass
class LoadingPoint:
    """装载点数据结构"""
    x: float                             # X 坐标
    y: float                             # Y 坐标
    # z: float                           # Z 坐标
    width: float                         # 可用区域宽度
    height: float                        # 可用区域高度
    active: bool = True                  # 是否可用

@dataclass
class Solution:
    """装载方案结果"""
    items:  List[Union[ProductsSpec, PalletSpec]]
    positions: List[Tuple[float, float, float, Tuple[float, float, float]]]   # (x, y, z, rotated_dimensions)
    volume_utilization: float = 0.0         # 体积利用率
    weight_utilization: float = 0.0         # 载重利用率
    stability_score: float = 0.0            # 重心偏移
    fitness: float = 0.0                    # 综合适应度
    placements: List[Placement] = field(default_factory=list)   # 货物放置信息

    def __post_init__(self):
        """在初始化后计算适应度"""
        self.fitness = self._calculate_fitness()

    def _calculate_fitness(self) -> float:
        """综合三个指标计算适应度（加权求和）"""
        weights = {
            'volume': 0.5,    # 体积利用率权重
            'weight': 0.3,    # 载重利用率权重
            'stability': 0.2  # 稳定性权重
        }
        return (
            weights['volume'] * self.volume_utilization +
            weights['weight'] * self.weight_utilization +
            weights['stability'] * self.stability_score
        )

@dataclass
class Placement:
    """货物放置信息"""
    product: ProductsSpec
    position: Tuple[float, float, float]        # (x, y, z)
    dimensions: Tuple[float, float, float]      # (l, w, h)
    pallet_id: int = 0

@dataclass
class Particle:
    """粒子群优化粒子"""
    solution: Solution = field(default_factory=lambda: Solution(items=[], positions=[]))         # 关联的装载方案
    position: List[float] = field(default_factory=list)                                          # 当前位置(优化变量)
    velocity: List[float] = field(default_factory=list)                                          # 当前速度
    best_position: List[float] = field(default_factory=list)                                     # 个体历史最优位置
    best_fitness: float = 0.0                                                                    # 个体历史最优适应度
    current_fitness: float = 0.0
    best_solution: Solution = field(default_factory=lambda: Solution(items=[], positions=[]))    # 最优解决方案

    @classmethod
    def generate_random(cls, solution: Solution) -> 'Particle':
        """生成随机粒子"""
        if not solution.items:
            return cls(
                solution=Solution(),
                position=[],
                velocity=[],
                best_position=[],
                best_fitness=0.0,
                best_solution=Solution()
            )
        
        # 生成随机位置和速度
        position = [random.uniform(0, 1) for _ in range(len(solution.items))]
        velocity = [0.0 for _ in position]
        
        return cls(
            solution=deepcopy(solution),
            position=position,
            velocity=velocity,
            best_position=deepcopy(position),
            best_fitness=solution.fitness,
            best_solution=deepcopy(solution)  # 初始最佳解就是当前解
        )

    def copy(self) -> 'Particle':
        """深拷贝粒子"""
        return Particle(
            solution=deepcopy(self.solution),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness,
            best_solution=deepcopy(self.best_solution)
        )