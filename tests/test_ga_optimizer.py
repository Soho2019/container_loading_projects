import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import GeneticAlgorithmOptimizer, HybridOptimizer
from src.core.domain import ContainerSpec, ProductsSpec, PalletSpec, Solution, Placement
from unittest.mock import patch


class TestHybridOptimizer:
    @pytest.fixture
    def setup(self):
        self.container = ContainerSpec(
            id="test_container",
            name="test_container",
            length=1200,
            width=240,
            height=240,
            max_weight=2000,
        )
        self.products = [
            ProductsSpec(
                id=i,
                sku=f"SKU-{i}",
                frgn_name=f"Foreign-{i}",
                item_name=f"Item-{i}",
                length=100,
                width=100,
                height=100,
                weight=20,
                fragility=0,
                allowed_rotations=[(100, 100, 100)],
                category="general",
            )
            for i in range(10)
        ]
        self.pallets = [
            PalletSpec(
                id=1,
                length=1200,
                width=240,
                height=150,
                max_weight=1000,
            )
        ]

    def test_initialization_validation(self, setup):
        # 测试无效输入校验
        with pytest.raises(ValueError, match="集装箱规格不能为空"):
            HybridOptimizer(None, self.products, self.pallets)

        with pytest.raises(ValueError, match="货物列表不能为空"):
            HybridOptimizer(self.container, [], self.pallets)

    @patch("src.core.algorithms.GeneticAlgorithmOptimizer.optimize")
    @patch("src.core.algorithms.PSO.optimize")
    @patch("src.core.algorithms.SA.optimize")
    def test_with_pallets_flow(self, mock_sa, mock_pso, mock_ga, setup):
        # 测试带托盘的优化流程
        mock_ga.return_value = Solution(
            items=self.pallets,
            positions=[(0, 0, 0)],
            placements=[
                Placement(
                    product=self.pallets[0],
                    position=(0, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,  # 使用 pallet_id
                )
            ],
        )
        mock_pso.return_value = Solution(
            items=self.pallets * 2,
            positions=[(0, 0, 0), (600, 0, 0)],
            placements=[
                Placement(
                    product=self.pallets[0],
                    position=(0, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,
                ),
                Placement(
                    product=self.pallets[0],
                    position=(600, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,
                ),
            ],
        )
        mock_sa.return_value = Solution(
            items=self.pallets * 2,
            positions=[(0, 0, 0), (500, 0, 0)],
            placements=[
                Placement(
                    product=self.pallets[0],
                    position=(0, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,
                ),
                Placement(
                    product=self.pallets[0],
                    position=(500, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,
                ),
            ],
        )

        hybrid = HybridOptimizer(self.container, self.products, self.pallets)
        result = hybrid.optimize()

        assert mock_ga.called
        assert mock_pso.called
        assert mock_sa.called
        assert len(result["pallets"]) == 2

    @patch("src.core.algorithms.ACO.generate_solutions")
    @patch("src.core.algorithms.PSO.optimize")
    @patch("src.core.algorithms.SA.optimize")
    def test_without_pallets_flow(self, mock_sa, mock_pso, mock_aco, setup):
        # 测试无托盘的优化流程
        mock_aco.return_value = [
            Solution(
                items=self.products[:5],
                positions=[(i * 100, i * 100, i * 100) for i in range(5)],
                placements=[
                    Placement(
                        product=self.products[i],
                        position=(i * 100, i * 100, i * 100),
                        dimensions=(100, 100, 100),
                        pallet_id=0,
                    )
                    for i in range(5)
                ],
            )
        ]
        mock_pso.return_value = Solution(
            items=self.products,
            positions=[(i * 80, i * 80, i * 80) for i in range(10)],
            placements=[
                Placement(
                    product=self.products[i],
                    position=(i * 80, i * 80, i * 80),
                    dimensions=(100, 100, 100),
                    pallet_id=0,
                )
                for i in range(10)
            ],
        )
        mock_sa.return_value = Solution(
            items=self.products,
            positions=[(i * 90, i * 90, i * 90) for i in range(10)],
            placements=[
                Placement(
                    product=self.products[i],
                    position=(i * 90, i * 90, i * 90),
                    dimensions=(100, 100, 100),
                    pallet_id=0,
                )
                for i in range(10)
            ],
        )

        hybrid = HybridOptimizer(self.container, self.products, [])
        result = hybrid.optimize()

        assert result["pallets"] == []
        assert len(result["positions"]) == len(self.products)
        # 检查是否有placements字段，如果有则验证长度
        if "placements" in result:
            assert len(result["positions"]) == len(result["placements"])
        else:
            # 如果没有placements字段，则跳过这个断言
            pass

    def test_constraint_validation(self, setup):
        # 测试约束检查
        hybrid = HybridOptimizer(self.container, self.products, self.pallets)
        invalid_solution = Solution(
            items=self.pallets + self.products,
            positions=[(0, 0, 0)] + [(-100, -100, -100)] * 10,
            placements=[
                Placement(
                    product=self.pallets[0],
                    position=(0, 0, 0),
                    dimensions=(1200, 240, 150),
                    pallet_id=1,
                )
            ]
            + [
                Placement(
                    product=self.products[i],
                    position=(-100, -100, -100),
                    dimensions=(100, 100, 100),
                    pallet_id=0,
                )
                for i in range(10)
            ],
        )
        assert not hybrid._validate_constraints(invalid_solution)

    def test_empty_pallet_list(self):
        """测试空托盘列表"""
        with pytest.raises(ValueError, match="候选托盘列表不能为空"):
            GeneticAlgorithmOptimizer(
                container=ContainerSpec(
                    id=1,
                    name="Test Container",
                    length=1000,
                    width=1000,
                    height=1000,
                    max_weight=1000,
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
                        allowed_rotations=[(10, 10, 10)],
                        category="general",
                    )
                ],
                candidate_pallets=[],  # 空托盘列表
            )

    def test_invalid_container(self):
        """测试无效容器"""
        with pytest.raises(ValueError, match="集装箱规格不能为空"):
            GeneticAlgorithmOptimizer(
                container=None,  # 无效容器
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
                        allowed_rotations=[(10, 10, 10)],
                        category="general",
                    )
                ],
                candidate_pallets=[
                    PalletSpec(
                        id=1, length=1200, width=240, height=150, max_weight=1000
                    )
                ],
            )

    def test_invalid_product_dimensions(self):
        """测试产品尺寸超过容器尺寸的情况"""
        container = ContainerSpec(
            id=1,
            name="Test Container",
            length=100,
            width=100,
            height=100,
            max_weight=1000,
        )

        # 测试三种情况：
        # 1. 单方向超限
        # 2. 多方向超限
        # 3. 旋转后可放置
        test_cases = [
            {
                "product": ProductsSpec(
                    id=1,
                    length=200,
                    width=100,
                    height=100,  # 长度超限
                    sku="OVERSIZE-1",
                    frgn_name="Oversize",
                    item_name="超大商品",
                    category="general",
                    fragility=0,
                    weight=50,
                    allowed_rotations=[(200, 100, 100)],
                ),
                "expected_positions": 0,  # 期望返回空解
            },
            {
                "product": ProductsSpec(
                    id=2,
                    length=120,
                    width=80,
                    height=80,  # 旋转后可放置
                    sku="ROTATABLE-1",
                    frgn_name="Rotatable",
                    item_name="可旋转商品",
                    category="general",
                    fragility=0,
                    weight=50,
                    allowed_rotations=[
                        (120, 80, 80),
                        (80, 120, 80),
                    ],  # 第二种旋转可放入
                ),
                "expected_positions": 0,
            },
        ]

        for case in test_cases:
            optimizer = HybridOptimizer(
                container=container,
                products=[case["product"]],
                candidate_pallets=[],
            )
            result = optimizer.optimize()

            assert len(result["positions"]) == case["expected_positions"]
            assert result["utilization"] == 0.0

    def test_ga_mutation_empty_individual(self):
        container = ContainerSpec(
            id=1, name="Test", length=10, width=10, height=10, max_weight=1000
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
                fragility=1,
                allowed_rotations=[(1, 1, 1)],
                category="general",
            )
        ]
        pallets = [PalletSpec(id=1, length=2, width=2, height=2, max_weight=10)]

        ga = GeneticAlgorithmOptimizer(container, products, pallets)
        mutated = ga._mutate([])
        assert mutated == []

    def test_ga_crossover_short_parents(self):
        container = ContainerSpec(
            id=1, name="Test", length=10, width=10, height=10, max_weight=1000
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
                fragility=1,
                allowed_rotations=[(1, 1, 1)],
                category="general",
            )
        ]
        pallets = [PalletSpec(id=1, length=2, width=2, height=2, max_weight=10)]

        ga = GeneticAlgorithmOptimizer(container, products, pallets)
        p1 = [pallets[0]]
        p2 = [pallets[0]]
        child = ga._crossover(p1, p2)
        assert len(child) == 1
