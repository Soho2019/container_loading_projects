"""
数据库模型文件

用于定义数据库模型，使用 SQLAlchemy ORM 来描述各个数据表的结构。

包含以下模型类：
- Product：表示产品信息，包括产品 ID、SKU、名称、尺寸、重量等属性。
- Pallet：表示托盘信息，包含托盘 ID、尺寸和最大承重等。
- Container：表示集装箱信息，包含集装箱 ID、名称、尺寸和最大承重等。
- HistoryFile：表示历史文件信息，有文件 ID、用户 ID、方案 ID、文件名和路径等。
- LoadingScheme：表示装载方案信息，包括方案 ID、用户 ID、容器 ID 和方案数据等。
- User：表示用户信息，如用户 ID、用户名、密码、邮箱和角色等。

每个模型类都定义了相应的字段、数据类型、约束条件以及默认值。
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    DECIMAL,
    ForeignKey,
    Text,
    CheckConstraint,
)
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from sqlalchemy import func
from database.base import Base

Base = declarative_base()


class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(50), unique=True, nullable=False)
    frgn_name = Column(String(100))
    item_name = Column(String(100))
    length = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    weight = Column(DECIMAL(10, 3), nullable=False)
    direction = Column(Integer, CheckConstraint("direction IN (0, 1)"), default=0)
    fragile = Column(Integer, CheckConstraint("fragile IN (0, 1, 2, 3)"), default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Pallet(Base):
    __tablename__ = "pallets"
    pallet_id = Column(Integer, primary_key=True, autoincrement=True)
    length = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    max_weight = Column(DECIMAL(10, 3), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Container(Base):
    __tablename__ = "containers"
    container_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    length = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    max_weight = Column(DECIMAL(10, 3), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    reports = relationship("PackingReport", back_populates="container")


class HistoryFile(Base):
    __tablename__ = "history_files"
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    scheme_id = Column(Integer, ForeignKey("loading_schemes.scheme_id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LoadingScheme(Base):
    __tablename__ = "loading_schemes"
    scheme_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    container_id = Column(
        Integer, ForeignKey("containers.container_id"), nullable=False
    )
    scheme_data = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(100))
    role = Column(Integer, CheckConstraint("role IN (0, 1, 2)"), default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PackingReport(Base):
    __tablename__ = "packing_reports"

    report_id = Column(Integer, primary_key=True)
    solution_id = Column(String(50), unique=True, nullable=False)
    container_id = Column(Integer, ForeignKey("containers.container_id"))
    creation_date = Column(DateTime, default=datetime.now)
    total_items = Column(Integer)
    total_volume = Column(Float)  # in m³
    total_weight = Column(Float)  # in kg
    utilization = Column(Float)  # 0-1
    stability = Column(Float)  # 0-1
    notes = Column(Text)
    image_path = Column(String(255))  # 缩略图路径

    # 关系
    container = relationship("Container", back_populates="reports")

    def __repr__(self):
        return f"<PackingReport(solution_id='{self.solution_id}', date='{self.creation_date}')>"
