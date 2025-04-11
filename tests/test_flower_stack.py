"""
花垛算法测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import FlowerStackOptimizer
from src.core.domain import ContainerSpec, ProductsSpec, Placement
from pulp import LpStatus


class TestFlowerStack:
    @pytest.fixture
    def setup(self):
        self.container = ContainerSpec(
            id="test_container",
            name="test",
            length=1200,
            width=240,
            height=240,
            max_weight=1000,  # 假设最大承重为100000
        )
        self.products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name=f"Foreign-{i}",
                item_name=f"Item-{i}",
                length=80 + i * 10,
                width=80 - i * 5,
                height=50,
                weight=20,
                fragility=i % 2,
                allowed_rotations=[0, 90],  # 假设允许0度和90度旋转
                category="general",
            )
            for i in range(10)
        ]
        self.optimizer = FlowerStackOptimizer(self.container, self.products)

    def test_layer_building(self, setup):
        layer = self.optimizer._build_layer()
        assert len(layer) > 0
        assert all(isinstance(p, Placement) for p in layer)

        # 验证层宽度不超过容器
        total_width = sum(p.dimensions[0] for p in layer)
        assert total_width <= self.container.width

    def test_lp_selection(self, setup):
        selected = self.optimizer._select_by_lp(500)  # 最大宽度500
        assert len(selected) <= len(self.products)

        # 验证总宽度限制
        if selected:
            total_width = sum(p.width for p in selected)
            assert total_width <= 500

    def test_centroid_adjustment(self, setup):
        # 创建测试层
        layer = [
            Placement(
                product=self.products[0],  # weight=20
                position=(0, 0, 0),
                dimensions=(100, 100, 50),
            ),
            Placement(
                product=self.products[1],  # weight=20
                position=(100, 0, 0),
                dimensions=(100, 100, 50),
            ),
        ]

        # 强制设置前一层的跨度 (50-250)
        self.optimizer.layers = [
            [
                Placement(
                    product=self.products[2],
                    position=(50, 0, 0),
                    dimensions=(200, 100, 50),
                )
            ]
        ]

        self.optimizer._adjust_centroid(layer)

        # 验证质心是否在支撑范围内
        # 原始质心在 (0*20 + 100*20)/40 = 50
        # 前一层的支撑范围是 50-250，所以不需要调整
        assert layer[0].position[0] == 0  # 应该保持不变
        assert layer[1].position[0] == 100
