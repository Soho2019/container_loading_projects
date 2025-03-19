"""
此文件定义与算法交互的领域模型

如集装箱类、货物类、托盘类等

长度单位为毫米(mm), 重量单位为千克(kg)
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional

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

@dataclass
class PalletSpec:
    """托盘规格"""
    id: int
    length: float
    width: float
    height: float
    max_weight: float

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
    items: List[ProductsSpec]
    positions: List[Tuple[float, float, float]]
    volume_utilization: float
    weight_utilization: float
    stability_score: float = 0.0


@dataclass
class Placement:
    """货物放置信息"""
    product: 'ProductsSpec'
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]

@dataclass
class Particle:
    """粒子群优化粒子"""
    position: List[float]
    velocity: List[float] 
    best_position: List[float]
    best_fitness: float = -float('inf')

    @classmethod
    def generate_random(cls):
        return cls(
            position=np.random.rand(3).tolist(),
            velocity=np.zeros(3).tolist(),
            best_position=[]
        )

    def copy(self):
        return Particle(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness
        )