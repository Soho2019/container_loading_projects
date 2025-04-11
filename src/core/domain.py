"""
此文件定义与算法交互的领域模型

如集装箱类、货物类、托盘类等

长度单位为毫米(mm), 重量单位为千克(kg)
"""

from __future__ import annotations  # 支持类型注解的前向引用
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
    door_reserve: float = 50  # 门预留空间

    @property
    def volume(self) -> float:
        """计算体积(mm³)"""
        return self.length * self.width * self.height  # 体积


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
    fragility: int  # 易碎等级
    allowed_rotations: List[Tuple[int, int, int]]  # 允许的旋转姿态
    category: str  # 货物类别

    @property
    def volume(self) -> float:
        """计算体积(mm³)"""
        return self.length * self.width * self.height  # 体积

    def __hash__(self):
        return hash(self.id)


@dataclass
class PalletSpec:
    """托盘规格"""

    id: int
    length: float
    width: float
    height: float
    max_weight: float

    @property
    def volume(self) -> float:
        """计算体积(mm³)"""
        return self.length * self.width * self.height

    @property
    def base_area(self) -> float:
        """计算托盘底面积(mm²)"""
        return self.length * self.width

    @property
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_weight": self.max_weight,
            "volume": self.volume,
        }


@dataclass
class LoadingPoint:
    """装载点数据结构"""

    x: float  # X 坐标
    y: float  # Y 坐标
    # z: float                           # Z 坐标
    width: float  # 可用区域宽度
    height: float  # 可用区域高度
    active: bool = True  # 是否可用


@dataclass
class Solution:
    """装载方案结果"""

    items: List[Union[ProductsSpec, PalletSpec]]
    positions: List[Tuple]  # 每个元素为(x, y, z) 或 (x, y, z, (l, w, h))
    volume_utilization: float = 0.0  # 体积利用率
    weight_utilization: float = 0.0  # 载重利用率
    stability_score: float = 0.0  # 重心偏移
    fitness: float = 0.0  # 综合适应度
    placements: List[Placement] = field(default_factory=list)  # 货物放置信息

    def __post_init__(self):
        """验证输入并计算适应度"""
        # 类型验证
        if not isinstance(self.items, list):
            raise TypeError("items必须是列表")
        if not isinstance(self.positions, list):
            raise TypeError("positions必须是列表")
        if not isinstance(self.placements, list):
            raise TypeError("placements必须是列表")

        # 长度一致性验证
        if len(self.items) != len(self.positions):
            raise ValueError("items和positions长度必须一致")
        if self.placements and len(self.placements) != len(self.items):
            raise ValueError("placements长度必须与items一致")
        if len(self.positions) != len(self.placements):
            raise ValueError("positions 和 placements 长度不一致")

        # 计算适应度
        self._calculate_fitness()

        # 自动填充placements如果为空
        if not self.placements and self.items and self.positions:
            self._generate_placements()

    def _calculate_fitness(self):
        """计算综合适应度"""
        self.fitness = (
            0.5 * self.volume_utilization
            + 0.3 * self.weight_utilization
            + 0.2 * self.stability_score
        )

    def _generate_placements(self):
        """根据items和positions自动生成placements"""
        self.placements = []
        for item, pos in zip(self.items, self.positions):
            if len(pos) == 3:  # (x,y,z)
                dimensions = (item.length, item.width, item.height)
            else:  # (x,y,z,(l,w,h))
                dimensions = pos[3]

            self.placements.append(
                Placement(
                    product=item if isinstance(item, ProductsSpec) else None,
                    pallet=item if isinstance(item, PalletSpec) else None,
                    position=pos[:3],
                    dimensions=dimensions,
                )
            )

    def __repr__(self):
        return (
            f"Solution(module={self.__class__.__module__}, "
            f"items={len(self.items)}, positions={len(self.positions)})"
        )


@dataclass
class Placement:
    """货物放置信息"""

    product: ProductsSpec
    position: Tuple[float, float, float]  # (x, y, z)
    dimensions: Tuple[float, float, float]  # (l, w, h)
    pallet_id: int = 0


@dataclass
class Particle:
    """粒子群优化粒子"""

    solution: Solution = field(
        default_factory=lambda: Solution(items=[], positions=[])
    )  # 关联的装载方案
    position: List[float] = field(default_factory=list)  # 当前位置(优化变量)
    velocity: List[float] = field(default_factory=list)  # 当前速度
    best_position: List[float] = field(default_factory=list)  # 个体历史最优位置
    best_fitness: float = 0.0  # 个体历史最优适应度
    current_fitness: float = 0.0
    best_solution: Solution = field(
        default_factory=lambda: Solution(items=[], positions=[])
    )  # 最优解决方案

    @classmethod
    def generate_random(cls, solution: Solution) -> "Particle":
        """生成随机粒子"""
        if not solution.items:
            return cls(
                solution=Solution(),
                position=[],
                velocity=[],
                best_position=[],
                best_fitness=0.0,
                best_solution=Solution(),
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
            best_solution=deepcopy(solution),  # 初始最佳解就是当前解
        )

    def copy(self) -> "Particle":
        """深拷贝粒子"""
        return Particle(
            solution=deepcopy(self.solution),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness,
            best_solution=deepcopy(self.best_solution),
        )


@dataclass
class PalletSolution:
    """托盘布局方案"""

    pallets: List[PalletSpec]
    positions: List[Tuple[float, float, float]]  # 托盘在集装箱内的坐标(x, y, z)
    utilization: float = 0.0  # 托盘利用率
    stability_score: float = 0.0  # 稳定性评分
    fitness: float = 0.0  # 综合适应度

    def __post_init__(self):
        """在初始化后计算适应度"""
        self.fitness = 0.6 * self.utilization + 0.4 * self.stability_score

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "pallets": [p.to_dict() for p in self.pallets],
            "positions": self.positions,
            "utilization": self.utilization,
            "stability_score": self.stability_score,
            "fitness": self.fitness,
        }
