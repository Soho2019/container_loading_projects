"""
塔装载算法测试
"""

import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import TowerPackingAlgorithm
from src.core.domain import ContainerSpec, ProductsSpec, PalletSpec
from src.config.constants import BusinessRules


class TestTowerPacking:
    """测试塔装载算法"""

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
            door_reserve=50,
        )

        # 初始化测试产品 - 包含完整参数
        self.products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name=f"Foreign-{i}",
                item_name=f"商品-{i}",
                category="general",
                length=80 + i * 5,
                width=80 + i * 5,
                height=50,
                weight=10,
                fragility=i % 3,
                allowed_rotations=[(80 + i * 5, 80 + i * 5, 50)],
            )
            for i in range(10)
        ]

        # 初始化托盘规格
        self.pallet = PalletSpec(
            id=1, length=1200, width=240, height=150, max_weight=1000
        )

        return TowerPackingAlgorithm(
            container=self.container,
            products=deepcopy(self.products),
            pallet=self.pallet,
            transport_type="sea",
            cargo_type="pallet",
            container_type="general",
        )

    def test_initialization(self, setup):
        tower = setup
        assert (
            tower.max_height
            == BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER
        )
        assert tower.edge_gap == BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET

    def test_orientation_selection(self, setup):
        tower = setup
        product = deepcopy(self.products[0])  # 创建副本避免修改原始数据
        product.allowed_rotations = [(100, 50, 50), (50, 100, 50), (50, 50, 100)]
        valid = tower._get_valid_orientations(product)
        assert len(valid) == 3  # 应全部有效

        # 测试尺寸超限情况
        product.allowed_rotations.append((2000, 2000, 2000))
        assert len(tower._get_valid_orientations(product)) == 3

    def test_fragile_stack_limits(self, setup):
        tower = setup
        fragile_products = [
            ProductsSpec(
                id=i,
                sku=f"FRAGILE-{i}",
                frgn_name=f"Fragile-{i}",
                item_name=f"易碎商品-{i}",
                category="fragile",
                length=50,
                width=50,
                height=50,
                weight=5,
                fragility=1,
                allowed_rotations=[(50, 50, 50)],
            )
            for i in range(10)
        ]
        tower.products = deepcopy(fragile_products)
        result = tower.optimize()

        max_stack = BusinessRules.FRAGILE_STACK_LIMIT[1]
        assert max(result["fragile_counts"].values()) <= max_stack

    def test_stability_calculation(self, setup):
        tower = setup
        result = tower.optimize()
        assert "stability" in result
        assert result["stability"]["stable"] in (True, False)

    def test_extreme_case_handling(self, setup):
        """测试超大货物"""
        tower = TowerPackingAlgorithm(
            container=ContainerSpec(
                id=2,
                name="Small Container",
                length=100,
                width=100,
                height=100,
                max_weight=500,
                door_reserve=10,
            ),
            products=[
                ProductsSpec(
                    id=99,
                    sku="OVERSIZE-1",
                    frgn_name="Oversize Product",
                    item_name="超大商品",
                    category="oversize",
                    length=200,
                    width=200,
                    height=200,
                    weight=100,
                    fragility=0,
                    allowed_rotations=[(200, 200, 200)],
                )
            ],
            pallet=PalletSpec(id=2, length=100, width=100, height=50, max_weight=200),
            transport_type="sea",
            cargo_type="pallet",
            container_type="general",
        )
        result = tower.optimize()
        assert len(result["positions"]) == 0  # 应无法放置

    def test_height_rules(self, setup):
        """测试不同运输场景下的高度限制"""
        container = deepcopy(self.container)
        pallet = deepcopy(self.pallet)

        # 普柜排托海运
        tower = TowerPackingAlgorithm(
            container=container,
            products=deepcopy(self.products),
            pallet=deepcopy(self.pallet),
            transport_type="sea",
            cargo_type="pallet",
            container_type="general",
        )
        assert tower.max_height == 2100

        # 高柜排托海运
        tower = TowerPackingAlgorithm(
            container=container,
            products=deepcopy(self.products),
            pallet=deepcopy(self.pallet),
            transport_type="sea",
            cargo_type="pallet",
            container_type="high",
        )
        assert tower.max_height == 2500

        # 普柜散货海运
        tower = TowerPackingAlgorithm(
            container=container,
            products=deepcopy(self.products),
            pallet=None,
            transport_type="sea",
            cargo_type="bulk",
            container_type="general",
        )
        assert tower.max_height == 2250  # 使用业务规则中的散货高度限制

    def test_invalid_transport_type(self):
        """测试无效运输类型"""
        with pytest.raises(ValueError):
            TowerPackingAlgorithm(
                container=ContainerSpec(
                    id=1,
                    name="Test Container",
                    length=1200,
                    width=240,
                    height=240,
                    max_weight=2000,
                    door_reserve=50,
                ),
                products=[
                    ProductsSpec(
                        id=1,
                        sku="SKU-1",
                        frgn_name="Product",
                        item_name="商品",
                        length=100,
                        width=100,
                        height=100,
                        weight=10,
                        fragility=0,
                        allowed_rotations=[(100, 100, 100)],
                        category="general",
                    )
                ],
                pallet=PalletSpec(
                    id=1, length=1200, width=240, height=150, max_weight=1000
                ),
                transport_type="invalid",
            )

    def test_empty_product_list(self):
        """测试空产品列表"""
        algo = TowerPackingAlgorithm(
            container=ContainerSpec(
                id=1,
                name="Test Container",
                length=1200,
                width=240,
                height=240,
                max_weight=2000,
                door_reserve=50,
            ),
            products=[],
            pallet=PalletSpec(
                id=1, length=1200, width=240, height=150, max_weight=1000
            ),
            transport_type="sea",
            cargo_type="pallet",
            container_type="general",
        )
        result = algo.optimize()
        assert len(result["positions"]) == 0

    def test_fragile_stack_limit(self):
        container = ContainerSpec(
            id=1, name="Test", length=10, width=10, height=10, max_weight=1000
        )
        pallet = PalletSpec(id=1, length=8, width=8, height=1, max_weight=100)
        products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i+1}",
                frgn_name="Product",
                item_name="商品",
                length=2,
                width=2,
                height=2,
                weight=1,
                fragility=1,
                allowed_rotations=[(2, 2, 2)],
                category="general",
            )
            for i in range(3)
        ]

        tower = TowerPackingAlgorithm(container, products, pallet, "sea", "pallet")
        result = tower.optimize()

        # 验证放置数量不超过堆叠限制
        max_stack = BusinessRules.FRAGILE_STACK_LIMIT[1]
        assert len(result["positions"]) <= max_stack * len(
            products
        )  # 最多每类堆叠max_stack层

        # 验证实际堆叠层数
        if result["positions"]:
            fragile_counts = result.get("fragile_counts", {})
            for count in fragile_counts.values():
                assert count <= max_stack
