# test_data_generator.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from src.core.domain import ProductsSpec

def generate_random_products(num=50):
    """生成随机货物数据"""
    products = []
    for i in range(1, num+1):
        length = random.randint(200, 2000)
        width = random.randint(200, 1200)
        height = random.randint(150, 800)
        weight = random.randint(3, 100)
        fragility = random.randint(0, 3)
        
        # 生成允许的旋转方向（至少包含原始方向）
        allowed_rotations = [(length, width, height)]
        if random.random() > 0.3:  # 70%概率允许长宽互换
            allowed_rotations.append((width, length, height))
        
        sku = f"SKU-{i:04d}"
        frgn_name = f"Product_{i}"
        item_name = f"货物_{i}"
        
        products.append(
            ProductsSpec(
                id=i,
                sku=sku,
                frgn_name=frgn_name,
                item_name=item_name,
                length=length,
                width=width,
                height=height,
                weight=weight,
                fragility=fragility,
                allowed_rotations=allowed_rotations,
                category=random.choice(["electronics", "furniture", "fragile", "general"])
            )
        )
    return products