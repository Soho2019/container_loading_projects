import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
from copy import deepcopy
from unittest.mock import patch, MagicMock
from src.core.algorithms import (
    GeneticAlgorithmOptimizer,
    PSO,
    SA,
    ACO,
    HybridOptimizer,
    calculate_volume_utilization,
    calculate_weight_utilization,
    calculate_center_offset,
    ParallelEvaluator,
)
from src.core.domain import ContainerSpec, ProductsSpec, PalletSpec, Solution, Placement
from src.config.constants import AlgorithmParams, BusinessRules


# 测试数据准备
@pytest.fixture
def sample_container():
    return ContainerSpec(
        id="cont1",
        name="Test",
        length=1200,
        width=800,
        height=800,
        max_weight=1000,
        door_reserve=100,
    )


@pytest.fixture
def sample_products():
    return [
        ProductsSpec(
            id=1,
            sku="SKU001",
            frgn_name="Foreign Product 1",
            item_name="Product 1",
            length=200,
            width=150,
            height=100,
            weight=10,
            allowed_rotations=[(200, 150, 100)],
            fragility=1,
            category="A",
        ),
        ProductsSpec(
            id=2,
            sku="SKU002",
            frgn_name="Foreign Product 2",
            item_name="Product 2",
            length=300,
            width=200,
            height=150,
            weight=20,
            allowed_rotations=[(300, 200, 150)],
            fragility=2,
            category="B",
        ),
    ]


@pytest.fixture
def sample_pallets():
    return [
        PalletSpec(
            id=1,
            length=1000,
            width=800,
            height=BusinessRules.PALLET_HEIGHT,
            max_weight=500,
        ),
        PalletSpec(
            id=2,
            length=1100,
            width=800,
            height=BusinessRules.PALLET_HEIGHT,
            max_weight=600,
        ),
    ]


# 1. 错误输入处理测试
def test_invalid_inputs(sample_container, sample_products):
    # 测试无效容器
    with pytest.raises(ValueError):
        calculate_volume_utilization(sample_products, None, False)

    # 测试无效产品尺寸
    invalid_products = deepcopy(sample_products)
    invalid_products[0].length = -100
    with pytest.raises(ValueError):
        calculate_volume_utilization(invalid_products, sample_container, False)

    # 测试零体积容器
    zero_container = ContainerSpec(
        id="zero",
        name="Zero Container",
        length=0,
        width=0,
        height=0,
        max_weight=1000,
        door_reserve=0,
    )
    with pytest.raises(ZeroDivisionError):
        calculate_volume_utilization(sample_products, zero_container, False)


# 2. 遗传算法复杂操作测试
def test_ga_complex_operations(sample_container, sample_products, sample_pallets):
    ga = GeneticAlgorithmOptimizer(
        container=sample_container,
        products=sample_products,
        candidate_pallets=sample_pallets,
    )

    # 设置较高的变异率以确保变异发生
    ga.mutation_rate = 1.0  # 100% 变异率

    # 测试变异操作
    individual = [sample_pallets[0], sample_pallets[1]]
    mutated = ga._mutate(individual)
    assert len(mutated) == len(individual)

    # 检查是否发生了变异
    # 由于随机性，可能需要多次尝试确保变异发生
    for _ in range(10):  # 尝试最多10次
        mutated = ga._mutate(individual)
        if mutated != individual:
            break
    assert mutated != individual, "变异操作未能改变个体"

    # 测试交叉操作
    p1 = [sample_pallets[0], sample_pallets[1]]
    p2 = [sample_pallets[1], sample_pallets[0]]
    child = ga._crossover(p1, p2)
    assert len(child) == len(p1)

    # 测试适应度计算
    fitness = ga._fitness(individual)
    assert 0 <= fitness <= 1


# 3. 模拟退火接受准则测试
def test_sa_acceptance_criteria(sample_container, sample_products):
    sa = SA(container=sample_container, products=sample_products)

    # 设置初始温度
    sa.temp = 1000
    sa.k = 1.0  # 确保Boltzmann常数为1，便于计算

    # 测试接受更优解（新能量更低）
    assert sa._accept(200, 100) is True

    # 测试接受差解的概率（新能量更高）
    # 计算理论接受概率
    delta = 100 - 200  # current_energy - new_energy
    acceptance_prob = np.exp(delta / (sa.k * sa.temp))

    # 由于随机性，我们测试多次确保大致符合概率
    accepted = 0
    trials = 1000
    for _ in range(trials):
        if sa._accept(200, 100):
            accepted += 1

    # 检查接受率是否在理论值附近（允许±10%误差）
    expected_rate = acceptance_prob * 100
    actual_rate = (accepted / trials) * 100
    assert abs(actual_rate - expected_rate) < 10

    # 低温下应该更难接受差解
    sa.temp = 0.1
    assert sa._accept(100, 200) is False

    # 相同能量应该接受
    assert sa._accept(100, 100) is True


# 4. 粒子群优化边界处理测试
def test_pso_boundary_handling(sample_container, sample_products):
    pso = PSO(container=sample_container, products=sample_products)

    # 创建测试解
    placements = [
        Placement(
            product=sample_products[0], position=(0, 0, 0), dimensions=(200, 150, 100)
        ),
        Placement(
            product=sample_products[1], position=(200, 0, 0), dimensions=(300, 200, 150)
        ),
    ]
    solution = Solution(
        items=sample_products,
        positions=[(0, 0, 0, (200, 150, 100)), (200, 0, 0, (300, 200, 150))],
        placements=placements,
    )

    # 测试修复
    repaired = pso._repair_solution(solution, sample_container)
    assert len(repaired.placements) == len(solution.placements)
    for pos in repaired.positions:
        assert pos[0] + pos[3][0] <= sample_container.length
        assert pos[1] + pos[3][1] <= sample_container.width
        assert pos[2] + pos[3][2] <= sample_container.height


# 5. 并行评估器测试
def test_parallel_evaluator(sample_container, sample_products):
    evaluator = ParallelEvaluator(sample_container)

    placements = [
        Placement(
            product=sample_products[0], position=(0, 0, 0), dimensions=(200, 150, 100)
        ),
        Placement(
            product=sample_products[1], position=(200, 0, 0), dimensions=(300, 200, 150)
        ),
    ]
    solution = Solution(
        items=sample_products,
        positions=[(0, 0, 0, (200, 150, 100)), (200, 0, 0, (300, 200, 150))],
        placements=placements,
    )

    # 测试评估
    evaluated = evaluator._evaluate_single(solution)
    assert hasattr(evaluated, "volume_utilization")
    assert hasattr(evaluated, "weight_utilization")
    assert hasattr(evaluated, "stability_score")
    assert hasattr(evaluated, "fitness")


# 6. 混合优化器复杂场景测试
def test_hybrid_optimizer_complex_scenarios(
    sample_container, sample_products, sample_pallets
):
    optimizer = HybridOptimizer(
        container=sample_container,
        products=sample_products,
        candidate_pallets=sample_pallets,
    )

    # 测试无托盘场景
    with patch.object(optimizer, "_run_ga") as mock_ga, patch.object(
        optimizer, "_run_pso"
    ) as mock_pso, patch.object(optimizer, "_run_sa") as mock_sa:
        mock_ga.return_value = []
        mock_pso.return_value = Solution(items=[], positions=[], placements=[])
        mock_sa.return_value = Solution(items=[], positions=[], placements=[])

        result = optimizer.optimize()
        assert "pallets" in result
        assert "positions" in result


# 7. 花垛算法线性规划测试
def test_flower_stack_lp(sample_container, sample_products):
    from src.core.algorithms import FlowerStackOptimizer

    optimizer = FlowerStackOptimizer(sample_container, sample_products)

    # 测试线性规划选择
    selected = optimizer._select_by_lp(500)
    assert selected is not None
    if selected:  # 可能有空选择
        assert all(p.width <= 500 for p in selected)


# 8. 特殊规则处理器测试
def test_special_rule_enforcer(sample_container, sample_products):
    from src.core.algorithms import SpecialRuleEnforcer

    enforcer = SpecialRuleEnforcer(sample_container, sample_products)

    # 创建测试解
    placements = [
        Placement(
            product=sample_products[0], position=(0, 0, 0), dimensions=(200, 150, 100)
        ),
        Placement(
            product=sample_products[1], position=(200, 0, 0), dimensions=(300, 200, 150)
        ),
    ]
    solution = Solution(
        items=sample_products,
        positions=[(0, 0, 0, (200, 150, 100)), (200, 0, 0, (300, 200, 150))],
        placements=placements,
    )

    # 测试惩罚计算
    penalty = enforcer.calculate_penalty(solution)
    assert penalty >= 0


def test_empty_input_solutions(sample_container):
    """测试空产品列表应该引发ValueError"""
    with pytest.raises(ValueError, match="货物列表不能为空"):
        HybridOptimizer(container=sample_container, products=[], candidate_pallets=[])
