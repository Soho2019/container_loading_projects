"""
混合优化器测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.core.algorithms import GeneticAlgorithmOptimizer, HybridOptimizer
from src.core.domain import ContainerSpec, ProductsSpec, PalletSpec
from src.config.constants import AlgorithmParams


class TestGeneticAlgorithm:
    """测试遗传算法优化器"""

    @pytest.fixture
    def setup(self):
        self.container = ContainerSpec(
            id=1,
            name="Test Container",
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
        return GeneticAlgorithmOptimizer(self.container, self.products, self.pallets)

    def test_initialization(self, setup):
        ga = setup
        assert ga.pop_size == AlgorithmParams.TRAY_POP_SIZE
        assert len(ga.sorted_products) == len(self.products)
        assert ga.sorted_products[0].volume >= ga.sorted_products[-1].volume

    def test_individual_generation(self, setup):
        ga = setup
        individual = ga._generate_individual()
        assert isinstance(individual, list)
        assert all(isinstance(p, PalletSpec) for p in individual)

    def test_fitness_calculation(self, setup):
        ga = setup
        individual = [self.pallets[0]]  # 单个托盘
        fitness = ga._fitness(individual)
        assert 0 <= fitness <= 1

        # 测试空个体
        assert ga._fitness([]) == 0.0

    def test_crossover_operation(self, setup):
        ga = setup
        p1 = [self.pallets[0], self.pallets[0]]
        p2 = [self.pallets[0]]
        child = ga._crossover(p1, p2)
        assert len(child) in (1, 2)  # 可能继承任一父代长度

    def test_mutation_operation(self, setup):
        ga = setup
        # 添加更多候选托盘规格以支持变异
        ga.candidate_pallets = [
            PalletSpec(id=1, length=1200, width=240, height=150, max_weight=1000),
            PalletSpec(id=2, length=1100, width=220, height=150, max_weight=900),
            PalletSpec(id=3, length=1000, width=200, height=150, max_weight=800),
        ]

        original = [ga.candidate_pallets[0]]
        mutated = None

        # 多次尝试变异以确保产生变化
        for _ in range(10):
            mutated = ga._mutate(original.copy())
            if mutated != original:
                break

        assert mutated != original, "变异操作应产生不同的托盘组合"
        assert all(
            p in ga.candidate_pallets for p in mutated
        ), "变异后的托盘应在候选列表中"

    def test_full_optimization(self, setup):
        ga = setup
        solution = ga.optimize()
        assert hasattr(solution, "fitness")
        assert solution.fitness > 0

    def test_invalid_product_type(self):
        """测试无效产品类型"""
        with pytest.raises(ValueError):
            HybridOptimizer(
                container=ContainerSpec(
                    id=1,
                    name="Test Container",
                    length=1200,
                    width=240,
                    height=240,
                    max_weight=2000,
                ),
                products=["invalid"],  # 非ProductsSpec类型
                candidate_pallets=[
                    PalletSpec(
                        id=1, length=1200, width=240, height=150, max_weight=1000
                    )
                ],
            )
