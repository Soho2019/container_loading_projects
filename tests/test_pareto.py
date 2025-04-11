"""
ParetoFront测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import ParetoFront, dominates
from src.core.domain import Solution, ProductsSpec, ContainerSpec, Placement


class TestParetoFront:
    def setup_method(self):
        # 创建测试用的产品
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product-1",
                item_name="商品-1",
                length=100,
                width=100,
                height=100,
                weight=10,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
        ]

        # 创建测试用的解决方案
        self.solutions = [
            Solution(
                items=products,
                positions=[(0, 0, 0, (100, 100, 100))],
                placements=[
                    Placement(
                        product=products[0],
                        position=(0, 0, 0),
                        dimensions=(100, 100, 100),
                    )
                ],
                volume_utilization=0.8,
                weight_utilization=0.7,
                stability_score=0.9,
            ),
            Solution(
                items=products,
                positions=[(100, 0, 0, (100, 100, 100))],
                placements=[
                    Placement(
                        product=products[0],
                        position=(100, 0, 0),
                        dimensions=(100, 100, 100),
                    )
                ],
                volume_utilization=0.7,
                weight_utilization=0.8,
                stability_score=0.8,
            ),
            Solution(
                items=products,
                positions=[(200, 0, 0, (100, 100, 100))],
                placements=[
                    Placement(
                        product=products[0],
                        position=(200, 0, 0),
                        dimensions=(100, 100, 100),
                    )
                ],
                volume_utilization=0.9,
                weight_utilization=0.6,
                stability_score=0.7,
            ),
        ]

    def test_top_solutions(self):
        front = ParetoFront(self.solutions)
        top = front.top(2)
        assert len(top) == 2
        assert top[0].fitness >= top[1].fitness
        assert not dominates(top[1], top[0])  # 确保前一个解不被后一个解支配

    def test_dominance_relation(self):
        # 第一个解在三个指标上都优于第二个解
        assert dominates(self.solutions[0], self.solutions[1]) is False
        # 第一个解和第三个解互不支配
        assert dominates(self.solutions[0], self.solutions[2]) is False
        assert dominates(self.solutions[2], self.solutions[0]) is False

    def test_empty_front(self):
        front = ParetoFront([])
        assert len(front.top(1)) == 0
