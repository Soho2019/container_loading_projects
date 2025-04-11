"""
此文件用于实现核心约束列表

覆盖文档中提到的所有约束，包括：
    体积约束（集装箱容量限制）

    载重约束（集装箱最大载重）

    不重叠约束（货物不可交叉）

    不悬空约束（货物需有支撑）

    正交约束（货物摆放方向）

    边界约束（货物不越界）

    重心约束（重心在安全区）

    完全切割约束（堆叠块可被平面切割）

    易碎品堆叠限制

    托盘间隙约束
"""

from typing import List, Tuple, Dict
from scipy.spatial import KDTree
from config.constants import BusinessRules
from database.converters import decode_rotations
from database.models import Product, Pallet, Container

class ConstraintChecker:
    def __init__(self, container, use_pallet: bool):
        self.container = container
        self.use_pallet = use_pallet            # 是否使用托盘的标志
    
    def check_all(
            self, 
            products: List[Product], 
            pallets: List[Pallet], 
            product_positions: List[Tuple[float, float, float]], 
            pallet_positions: List[Tuple[float, float, float]]
            ) -> bool:
        """集成所有约束检查"""
        # 合并货物和托盘(如果使用托盘)
        all_items = products + pallets if self.use_pallet else products
        all_positions = product_positions + pallet_positions if self.use_pallet else product_positions

        return (
            self.check_volume(all_items) and
            self.check_weight(all_items) and
            self.check_overlap(all_items, all_positions) and
            self.check_suspension(products, product_positions, pallets, pallet_positions) and
            self.check_orientation(all_items) and
            self.check_boundary(all_items, all_positions) and
            self.check_centroid(all_items, all_positions) and
            self.check_cut(products, product_positions, pallets, pallet_positions) and
            self.check_fragility(products, product_positions) and
            self.check_tray_gap(pallets, pallet_positions, products, product_positions)
        )
    
    # ----------------- 工具函数 -----------------
    def _get_placed_dimensions(self, item: Product) -> Tuple[float, float, float]:
        """根据方向返回实际摆放尺寸"""
        if isinstance(item, Pallet):
            return (item.length, item.width, item.height)
        rotations = [
            (item.length, item.width, item.height),
            (item.length, item.height, item.width),
            (item.width, item.length, item.height),
            (item.width, item.height, item.length),
            (item.height, item.length, item.width),
            (item.height, item.width, item.length)
        ]
        return rotations[item.direction]
    
    # ----------------- 基础约束 -----------------
    def check_volume(self, products: List[Product]) -> bool:
        """体积约束, 货物+托盘总体积 ≤ 集装箱容量"""
        total_volume = sum(p.length * p.width * p.height for p in products)
        return total_volume <= self.container.length * self.container.width * self.container.height
    
    def check_weight(self, products: List[Product]) -> bool:
        """载重约束, 货物+托盘总重量 ≤ 集装箱最大载重 <= 集装箱最大载重"""
        total_weight = sum(p.weight for p in products)
        return total_weight <= self.container.max_weight
    
    # ----------------- 空间布局约束 -----------------
    def check_overlap(self, products: List[Product], positions: List[Tuple[float, float, float]]) -> bool:
        """不重叠约束, 货物不可交叉, AABB碰撞检测"""
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                xi, yi, zi = positions[i]
                li, wi, hi = products[i].length, products[i].width, products[i].height
                xj, yj, zj = positions[j]
                lj, wj, hj = products[j].length, products[j].width, products[j].height

                # 检查是否在x、y、z方向上重叠
                overlap_x = (xi < xj + lj) and (xi + li > xj)
                overlap_y = (yi < yj + wj) and (yi + wi > yj)
                overlap_z = (zi < zj + hj) and (zi + hi > zj)

                if overlap_x and overlap_y and overlap_z:
                    return False
        return True
    
    def check_suspension(
            self, 
            products: List[Product], 
            product_positions: List[Tuple[float, float, float]], 
            pallets: List[Pallet], 
            pallet_positions: List[Tuple[float, float, float]]
            ) -> bool:
        """不悬空约束, 货物底面需有支撑，支撑面积 95% 以上"""
        # 合并所有支撑物（托盘或货物）
        support_items = pallets + products if self.use_pallet else products
        support_positions = pallet_positions + product_positions if self.use_pallet else product_positions

        for (x, y, z), product in zip(product_positions, products):
            if z == 0:
                continue     # 货物直接放在集装箱底部, 无需检查

            bottom_area = product.length * product.width
            support_area = 0.0

            # 检查货物底部是否有支撑（货物或托盘）
            support_area = 0.0
            for (ox, oy, oz), other in zip(support_positions, support_items):
                if oz + other.height == z:          # 支撑物顶部与当前货物底部齐平
                    # 计算投影重叠面积
                    overlap_x = max(0, min(x + product.length, ox + other.length) - max(x, ox))
                    overlap_y = max(0, min(y + product.width, oy + other.width) - max(y, oy))
                    support_area += overlap_x * overlap_y
            
            # 检查支撑面积比例
            if support_area / bottom_area < BusinessRules.SUPPORT_AREA_MIN_LIMIT:
                return False
        return True
    
    # ----------------- 摆放规则约束 -----------------
    def check_orientation(self, products: List[Product]) -> bool:
        """正交约束, 验证货物是否按允许的方向摆放"""
        for product in products:
            allowed_rotations = decode_rotations(product.direction)
            actual_rotation = (
                product.length,
                product.width,
                product.height
            )
            if actual_rotation not in allowed_rotations:
                return False
        return True
    
    def check_boundary(self, products: List[Product], positions: List[Tuple[float, float, float]]) -> bool:
        """边界约束, 边界约束：货物/托盘不超出集装箱"""
        for (x, y, z), product in zip(positions, products):
            pl, pw, ph = self._get_placed_dimensions(product)
            if (x + pl > self.container.length or
                y + pw > self.container.width or
                z + ph > self.container.height):
                return False
        return True
    
    # ----------------- 重心与稳定性约束 -----------------
    def check_centroid(self, products: List[Product], positions: List[Tuple[float, float, float]]) -> bool:
        """重心约束, 货物重心在安全区内"""
        total_weight = sum(p.weight for p in products)
        if total_weight == 0:
            return True         # 无货物视为非法
        
        # 计算重心坐标
        weighted_x, weighted_y, weighted_z = 0.0, 0.0, 0.0
        for (x, y, z), product in zip(positions, products):
            pl, pw, ph = self._get_placed_dimensions(product)
            center_x = x + pl / 2
            center_y = y + pw / 2
            center_z = z + ph / 2
            weighted_x += center_x * product.weight
            weighted_y += center_y * product.weight
            weighted_z += center_z * product.weight

        center_gx = weighted_x / total_weight
        center_gy = weighted_y / total_weight
        center_gz = weighted_z / total_weight

        # 安全范围 (5%)
        x_min, x_max = self.container.length * 0.45, self.container.length * 0.55
        y_min, y_max = self.container.width * 0.45, self.container.width * 0.55
        z_min, z_max = self.container.height * 0.45, self.container.height * 0.55

        return (
            x_min <= center_gx <= x_max and
            y_min <= center_gy <= y_max and
            z_min <= center_gz <= z_max
        )
    
    def calculate_stability_score(self, products, positions) -> float:
        """计算稳定性得分(0-1)"""
        if not products:
            return 0.0
            
        total_weight = sum(p.weight for p in products)
        if total_weight <= 0:
            return 0.0
        
        weighted_x = sum((pos[0] + dim[0]/2) * p.weight 
                        for p, pos, dim in zip(products, positions, 
                                            [self._get_placed_dimensions(p) for p in products]))
        weighted_y = sum((pos[1] + dim[1]/2) * p.weight 
                        for p, pos, dim in zip(products, positions, 
                                            [self._get_placed_dimensions(p) for p in products]))
        
        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight
        
        offset_x = abs(center_x - self.container.length/2) / (self.container.length/2)
        offset_y = abs(center_y - self.container.width/2) / (self.container.width/2)
        
        return 1 - (offset_x + offset_y)/2
    
    # ----------------- 特殊业务规则约束 -----------------
    def check_cut(
            self, 
            products: List[Product], 
            product_positions: List[Tuple[float, float, float]], 
            pallets: List[Pallet], 
            pallet_positions: List[Tuple[float, float, float]]
            ) -> bool:
        """完全切割约束：托盘上的货物不得超出托盘边界"""
        # 检查托盘内货物是否越界
        for (px, py, pz), pallet in zip(pallet_positions, pallets):
            for (gx, gy, gz), product in zip(product_positions, products):
                pl, pw, ph = self._get_placed_dimensions(product)
                # 货物是否在该托盘上方
                if (gz >= pz + pallet.height and
                    gx >= px and gx + pl <= px + pallet.length and
                    gy >= py and gy + pw <= py + pallet.width):
                    continue
                else:
                    return False
        
        # 无托盘时，底层货物作为托盘， 检查约束
        if not self.use_pallet:
            bottom_layer = [pos for pos in product_positions if pos[2] == 0]
            for (gx, gy, gz), product in zip(product_positions, products):
                if gz > 0:      # 非底层货物
                    is_supported = False
                    for (bx, by, bz), base_product in zip(product_positions, products):
                        if bz == 0 and (bx <= gx <= bx + base_product.length and by <= gy <= by + base_product.width):
                            is_supported = True
                            break
                    if not is_supported:
                        return False
        return True


    def check_fragility(self, products: List[Product], positions: List[Tuple[float, float, float]]) -> bool:
        """易碎品堆叠限制"""
        stack_map = {}      # {(x, y): 当前位置堆叠层数}
        for (x, y, _), product in zip(positions, products):
            if product.fragile not in BusinessRules.FRAGILE_STACK_LIMIT:
                continue
            max_layers = BusinessRules.FRAGILE_STACK_LIMIT[product.fragile]
            key = (round(x, 2), round(y, 2))     # 避免浮点误差
            if stack_map.get(key, 0) >= max_layers:
                return False
            stack_map[key] = stack_map.get(key, 0) + 1
        return True
    
    def check_tray_gap(
            self, 
            pallets: List[Pallet], 
            pallet_positions: List[Tuple[float, float, float]], 
            products: List[Product], 
            product_positions: List[Tuple[float, float, float]]
            ) -> bool:
        """托盘间隙全约束检查"""
        if not self.use_pallet:
            return True
        
        # ----------------- 托盘与集装箱间隙 -----------------
        for (x, y, z), pallent in zip(pallet_positions, pallets):
            # 柜门方向（集装箱长度为x轴，柜门在尾部）
            if self.container.length - (x + pallent.length) < BusinessRules.PALLET_GAP_CONTAINER["back"]:
                return False
            # 左右间隙
            if y < BusinessRules.PALLET_GAP_CONTAINER["left"]:
                return False
            if self.container.width - (y + pallent.width) < BusinessRules.PALLET_GAP_CONTAINER["right"]:
                return False
            # 前间隙
            if y < BusinessRules.PALLET_GAP_CONTAINER["front"]:
                return False
            
        # ----------------- 托盘间间隙 -----------------
        for i in  range(len(pallets)):
            for j in range(i + 1, len(pallets)):
                xi, yi, _ =  pallet_positions[i]
                xj, yj, _ =  pallet_positions[j]
                li, wi = pallets[i].length, pallets[i].width
                lj, wj = pallets[j].length, pallets[j].width

                # 横向间隙（y）
                if abs(yi + wi - yj) < BusinessRules.PALLET_GAP["lateral"]:
                    return False
                # 纵向间隙(x)
                if abs(xi + li - xj) < BusinessRules.PALLET_GAP["longitudinal"]:
                    return False
        
        # ----------------- 货物与托盘边缘间隙 -----------------
        for (px, py, pz), pallent in zip(pallet_positions, pallets):
            for (gx, gy, gz), product in zip(product_positions, products):
                pl, pw, _ = self._get_placed_dimensions(product)
                # 检查货物是否在托盘上方
                if (gx >= px and gx + pl <= px + pallent.length and gy >= py and gy + pw <= py + pallent.width and gz >= pz):
                    # 货物与托盘边缘间隙
                    edge_gap = BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET
                    if (gx - px < edge_gap or gy - py < edge_gap or (px + pallent.length) - (gx + pl) < edge_gap or (py + pallent.width) - (gy + pw) < edge_gap):
                        return False
            return True
        