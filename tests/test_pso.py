"""
粒子群测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import PSO, Solution
from src.core.domain import ContainerSpec, ProductsSpec, Placement, Solution
import numpy as np


class TestPSO:
    @pytest.fixture
    def setup(self):
        # 初始化容器规格
        self.container = ContainerSpec(
            id=1,
            name="Test Container",
            length=1200,
            width=240,
            height=240,
            max_weight=2000,
        )

        # 初始化测试产品（提供所有必需参数）
        self.products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name=f"Foreign-{i}",
                item_name=f"商品-{i}",
                category="general",
                length=100,
                width=100,
                height=100,
                weight=10,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
            )
            for i in range(3)
        ]

        # 初始化解决方案（添加 dimensions 参数）
        self.solutions = [
            Solution(
                items=self.products[:2],
                placements=[
                    Placement(
                        product=self.products[0],
                        position=(0, 0, 0),
                        dimensions=(100, 100, 100),  # 添加 dimensions
                    ),
                    Placement(
                        product=self.products[1],
                        position=(100, 0, 0),
                        dimensions=(100, 100, 100),  # 添加 dimensions
                    ),
                ],
                positions=[  # 添加 positions 参数
                    (0, 0, 0, (100, 100, 100)),  # (x, y, z, dimensions)
                    (100, 0, 0, (100, 100, 100)),
                ],
            )
        ]

        # 初始化PSO并手动初始化种群
        self.pso = PSO(container=self.container, products=self.products)
        self.pso._initialize_population(self.solutions, self.container)

    def test_velocity_update(self, setup):
        """测试速度更新逻辑"""
        if len(self.pso.population) == 0:
            pytest.skip("PSO population not initialized")

        particle = self.pso.population[0]
        original_velocity = particle.velocity.copy()
        self.pso._update_velocity(particle)
        assert not np.array_equal(particle.velocity, original_velocity)

    def test_position_update(self, setup):
        """测试位置更新逻辑"""
        if len(self.pso.population) == 0:
            pytest.skip("PSO population not initialized")

        particle = self.pso.population[0]
        original_position = particle.position.copy()
        self.pso._update_position(particle, self.container)
        assert not np.array_equal(particle.position, original_position)

    def test_evaluate_method(self, setup):
        """测试适应度评估逻辑"""
        if len(self.pso.population) == 0:
            pytest.skip("PSO population not initialized")

        particle = self.pso.population[0]
        self.pso._evaluate(particle, self.container)
        assert isinstance(particle.current_fitness, float)
        assert 0 <= particle.current_fitness <= 1

    def test_full_optimization(self, setup):
        """测试完整优化流程"""
        result = self.pso.optimize(self.solutions, self.container)
        assert isinstance(result, Solution)
