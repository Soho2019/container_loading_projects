"""
NSGA-II测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import NSGAII, Solution, dominates
from src.core.domain import ProductsSpec, ContainerSpec, Solution, Placement


class TestNSGAII:
    @pytest.fixture
    def setup(self):
        self.nsga = NSGAII()

        # 创建测试用的产品和容器
        container = ContainerSpec(
            id=1,
            name="Test Container",
            length=1000,
            width=1000,
            height=1000,
            max_weight=1000,
        )

        # 创建测试用的产品列表
        products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name=f"Product-{i}",
                item_name=f"商品-{i}",
                length=100,
                width=100,
                height=100,
                weight=10,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
            for i in range(3)
        ]

        self.population = [
            Solution(
                items=products[:2],  # 使用前两个产品
                positions=[(0, 0, 0, (100, 100, 100)), (100, 0, 0, (100, 100, 100))],
                placements=[
                    Placement(
                        product=products[0],
                        position=(0, 0, 0),
                        dimensions=(100, 100, 100),
                    ),
                    Placement(
                        product=products[1],
                        position=(100, 0, 0),
                        dimensions=(100, 100, 100),
                    ),
                ],
                volume_utilization=0.7 + 0.1 * i,
                weight_utilization=0.6 - 0.05 * i,
                stability_score=0.8 + 0.02 * i,
            )
            for i in range(10)
        ]

    def test_fast_non_dominated_sort(self, setup):
        fronts = self.nsga._fast_non_domainated_sort(self.population)
        assert len(fronts) > 0
        assert all(len(front) > 0 for front in fronts)

    def test_crowding_distance(self, setup):
        front = self.population[:5]  # 取前5个作为测试front
        self.nsga._crowding_distance_assignment(front)
        assert hasattr(front[0], "crowding_distance")

    def test_evolutionary_operations(self, setup):
        parent1, parent2 = self.population[0], self.population[1]
        child = self.nsga._crossover(parent1, parent2)
        assert len(child.items) > 0

        mutated = self.nsga._mutate(child)
        assert len(mutated.items) > 0
