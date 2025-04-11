"""
二维装载测试
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.core.algorithms import BinPacking2D
from src.config.constants import BusinessRules


class TestBinPacking2D:
    """测试二维装载算法"""

    @pytest.fixture
    def setup(self):
        return BinPacking2D(container_size=(1200, 600))

    def test_initialization(self, setup):
        bin_pack = setup
        assert bin_pack.width == 1200
        assert bin_pack.height == 600
        assert len(bin_pack.points) == 1

    def test_greedy_packing(self, setup):
        bin_pack = setup
        items = [
            {"dimensions": (300, 300, 100), "id": 1},
            {"dimensions": (400, 400, 100), "id": 2},
            {"dimensions": (200, 200, 100), "id": 3},
        ]
        result = bin_pack.greedy_packing(items)
        assert len(result) == len(items)

    def test_hybrid_optimization(self, setup):
        bin_pack = setup
        items = [
            {"dimensions": (400, 400, 100), "fragility": 1, "id": 1},
            {"dimensions": (300, 300, 100), "fragility": 0, "id": 2},
            {"dimensions": (200, 200, 100), "fragility": 1, "id": 3},
            {"dimensions": (150, 300, 100), "fragility": 0, "id": 4},
            {"dimensions": (250, 150, 100), "fragility": 1, "id": 5},
            {"dimensions": (150, 150, 100), "fragility": 2, "id": 6},
            {"dimensions": (100, 100, 100), "fragility": 2, "id": 7},
        ]
        result = bin_pack.hybrid_optimize(items)
        assert result["utilization"] > 0.6
        assert len(result["positions"]) == len(items)

    def test_empty_container(self):
        """测试空容器应抛出异常"""
        with pytest.raises(ValueError, match="容器尺寸必须为正数"):
            BinPacking2D(container_size=(0, 0))

    def test_invalid_container_size(self):
        """测试无效容器尺寸"""
        with pytest.raises(ValueError, match="容器尺寸必须为正数"):
            BinPacking2D(container_size=(-100, -100))

        with pytest.raises(ValueError, match="容器尺寸必须为正数"):
            BinPacking2D(container_size=(100, -100))

    def test_duplicate_items(self):
        """测试重复物品"""
        bin_pack = BinPacking2D(container_size=(1200, 600))
        items = [{"dimensions": (300, 300, 100), "id": 1}] * 2
        result = bin_pack.greedy_packing(items)
        assert len(result) == 2
