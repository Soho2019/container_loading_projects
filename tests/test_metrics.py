"""
三大指标测试文件
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.algorithms import (
    calculate_volume_utilization,
    calculate_weight_utilization,
    calculate_center_offset,
)
from src.core.domain import ContainerSpec, ProductsSpec


class TestVolumeUtilization:
    """测试体积利用率计算"""

    @pytest.fixture
    def sample_products(self):
        return [
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
                allowed_rotations=[(10, 10, 10)],
                category="general",
            )
        ]

    @pytest.fixture
    def sample_container(self):
        return ContainerSpec(
            id=1,
            name="Test Container",
            length=1000,
            width=1000,
            height=1000,
            max_weight=1000,
        )

    @pytest.mark.parametrize(
        "use_pallet,expected_ratio",
        [
            (False, 0.001),  # 1000000/1000000000 = 0.001
            (
                True,
                0.001176,
            ),  # 1000000/(1000*1000*(1000-150)) ≈ 0.001176 (托盘高度150mm)
        ],
    )
    def test_basic_calculation(
        self, sample_products, sample_container, use_pallet, expected_ratio
    ):
        ratio = calculate_volume_utilization(
            sample_products, sample_container, use_pallet
        )
        assert abs(ratio - expected_ratio) < 0.0001

    def test_empty_container(self, sample_products):
        with pytest.raises(ZeroDivisionError):
            calculate_volume_utilization(
                sample_products,
                ContainerSpec(
                    id=1,
                    name="Empty",
                    length=0,
                    width=0,
                    height=0,
                    max_weight=0,
                ),
                False,
            )

    # 在TestVolumeUtilization类中添加
    def test_zero_products(self, sample_container):
        """测试空产品列表"""
        assert calculate_volume_utilization([], sample_container, False) == 0.0
        assert calculate_volume_utilization([], sample_container, True) == 0.0

    @pytest.mark.parametrize("dimension", ["length", "width", "height"])
    def test_negative_dimensions(self, dimension):
        """测试负尺寸产品"""
        product_data = {
            "id": 1,
            "sku": "SKU-1",
            "frgn_name": "Product",
            "item_name": "商品",
            "length": 100,
            "width": 100,
            "height": 100,
            "weight": 10,
            "fragility": 0,
            "allowed_rotations": [(10, 10, 10)],
            "category": "general",
        }
        product_data[dimension] = -100  # 设置一个维度为负值

        with pytest.raises(ValueError):
            calculate_volume_utilization(
                [ProductsSpec(**product_data)],
                ContainerSpec(
                    id=1,
                    name="Test",
                    length=1000,
                    width=1000,
                    height=1000,
                    max_weight=1000,
                ),
                False,
            )

    # 在TestWeightUtilization类中添加
    def test_zero_weight_products(self):
        """测试重量为0的产品"""
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=100,
                width=100,
                height=100,
                weight=0,
                fragility=0,
                allowed_rotations=[(10, 10, 10)],
                category="general",
            )
        ]
        container = ContainerSpec(
            id=1,
            name="Test",
            length=1000,
            width=1000,
            height=1000,
            max_weight=1000,
        )
        assert calculate_weight_utilization(products, container) == 0.0

    # 在TestCenterOffset类中添加
    def test_single_product(self):
        """测试单个产品的重心计算"""
        product = ProductsSpec(
            id=1,
            sku="SKU-1",
            frgn_name="Product",
            item_name="商品",
            length=10,
            width=10,
            height=10,
            weight=10,
            fragility=0,
            allowed_rotations=[(10, 10, 10)],
            category="general",
        )
        container = ContainerSpec(
            id=1,
            name="Test",
            length=100,
            width=100,
            height=100,
            max_weight=10000,
        )
        positions = [(0, 0, 0, (10, 10, 10))]
        offsets = calculate_center_offset([product], container, positions)
        assert offsets == (40.0, 40.0, 40.0)  # 单个产品的重心就是其几何中心

    def test_volume_utilization_zero_container(self):
        container = ContainerSpec(
            id=1, name="Test", length=0, width=10, height=10, max_weight=1000
        )
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=1,
                width=1,
                height=1,
                weight=1,
                fragility=0,
                allowed_rotations=[(1, 1, 1)],
                category="general",
            )
        ]

        with pytest.raises(ZeroDivisionError):
            calculate_volume_utilization(products, container, False)

    def test_volume_utilization_invalid_product(self):
        container = ContainerSpec(
            id=1, name="Test", length=10, width=10, height=10, max_weight=1000
        )
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=0,
                width=1,
                height=1,
                weight=1,
                fragility=0,
                allowed_rotations=[(0, 1, 1), (1, 0, 1), (1, 1, 0)],
                category="general",
            )
        ]

        with pytest.raises(ValueError):
            calculate_volume_utilization(products, container, False)


class TestWeightUtilization:
    """测试重量利用率计算"""

    @pytest.fixture
    def sample_products(self):
        return [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name="Product",
                item_name="商品",
                length=100,
                width=100,
                height=100,
                weight=500,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
            for i in range(2)
        ]

    def test_normal_case(self, sample_products):
        container = ContainerSpec(
            id=1, name="Test", max_weight=1000, length=1000, width=1000, height=1000
        )
        assert calculate_weight_utilization(sample_products, container) == 1.0

    def test_overweight_scenario(self):
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=100,
                width=100,
                height=100,
                weight=600,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
        ]
        with pytest.raises(ValueError, match="载重超限"):
            calculate_weight_utilization(
                products,
                ContainerSpec(
                    id=1,
                    name="Test",
                    max_weight=500,
                    length=1000,
                    width=1000,
                    height=1000,
                ),
            )

    @pytest.mark.parametrize("max_weight,expected", [(0, 0.0), (-100, 0.0)])
    def test_edge_cases(self, max_weight, expected):
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=100,
                width=100,
                height=100,
                weight=100,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
        ]
        assert (
            calculate_weight_utilization(
                products,
                ContainerSpec(
                    id=1,
                    name="Test",
                    max_weight=max_weight,
                    length=1000,
                    width=1000,
                    height=1000,
                ),
            )
            == expected
        )

    @pytest.mark.parametrize("dimension", ["length", "width", "height"])
    def test_invalid_product_dimensions(self, dimension):
        """测试产品尺寸为0或负数的情况"""
        product_data = {
            "id": 1,
            "sku": "SKU-1",
            "frgn_name": "Product",
            "item_name": "商品",
            "length": 100,
            "width": 100,
            "height": 100,
            "weight": 10,
            "fragility": 0,
            "allowed_rotations": [(10, 10, 10)],
            "category": "general",
        }
        product_data[dimension] = 0  # 设置一个维度为0

        with pytest.raises(ValueError):
            calculate_volume_utilization(
                [ProductsSpec(**product_data)],
                ContainerSpec(
                    id=1,
                    name="Test",
                    length=1000,
                    width=1000,
                    height=1000,
                    max_weight=1000,
                ),
                False,
            )

    def test_weight_utilization_overload(self):
        container = ContainerSpec(
            id=1, name="Test", length=10, width=10, height=10, max_weight=1
        )
        products = [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=1,
                width=1,
                height=1,
                weight=2,
                fragility=0,
                allowed_rotations=[(1, 1, 1)],
                category="general",
            )
        ]

        with pytest.raises(ValueError):
            calculate_weight_utilization(products, container)


class TestCenterOffset:
    """测试重心偏移计算"""

    @pytest.fixture
    def container(self):
        return ContainerSpec(
            id=1, name="Test", length=100, width=100, height=100, max_weight=10000
        )

    @pytest.fixture
    def products(self):
        return [
            ProductsSpec(
                id=1,
                sku="SKU-1",
                frgn_name="Product",
                item_name="商品",
                length=10,
                width=10,
                height=10,
                weight=10,
                fragility=0,
                allowed_rotations=[(10, 10, 10)],
                category="general",
            ),
            ProductsSpec(
                id=2,
                sku="SKU-2",
                frgn_name="Product",
                item_name="商品",
                length=20,
                width=20,
                height=20,
                weight=20,
                fragility=0,
                allowed_rotations=[(20, 20, 20)],
                category="general",
            ),
        ]

    def test_balanced_load(self, products, container):
        positions = [(0, 0, 0, (10, 10, 10)), (50, 50, 50, (20, 20, 20))]
        offsets = calculate_center_offset(products, container, positions)
        assert all(0 <= o <= 50 for o in offsets)

    def test_unbalanced_load(self, products, container):
        positions = [(0, 0, 0, (10, 10, 10)), (0, 0, 0, (20, 20, 20))]
        total_offset = calculate_center_offset(products, container, positions, True)
        assert total_offset > 20

    def test_empty_input(self, container):
        assert calculate_center_offset([], container, []) == (float("inf"),) * 3

    def test_rotated_product_center(self):
        """测试旋转后的产品重心计算"""
        product = ProductsSpec(
            id=1,
            sku="SKU-1",
            frgn_name="Test Product",
            item_name="测试商品",
            category="general",
            fragility=0,
            length=20,
            width=10,
            height=5,
            weight=10,
            allowed_rotations=[(20, 10, 5), (10, 20, 5)],
        )

        # 测试不同旋转方向下的重心计算
        container = ContainerSpec(
            id=1,
            name="Test Container",
            length=100,
            width=100,
            height=100,
            max_weight=1000,
        )

        # 测试原始方向 (20,10,5)
        pos1 = (0, 0, 0, (20, 10, 5))
        offsets1 = calculate_center_offset([product], container, [pos1])
        assert offsets1 == (35.0, 40.0, 42.5)  # (x+l/2, y+w/2, z+h/2)

        # 测试旋转90度后的方向 (10,20,5)
        pos2 = (0, 0, 0, (10, 20, 5))
        offsets2 = calculate_center_offset([product], container, [pos2])
        assert offsets2 == (40.0, 35.0, 42.5)

        assert offsets1 != offsets2
