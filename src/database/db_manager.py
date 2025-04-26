"""
数据库管理文件

用于管理数据库连接和表的创建。

属性:
    engine: SQLAlchemy 创建的数据库引擎实例。
    Session: 绑定到数据库引擎的会话工厂。

方法:
    __init__(self, db_name): 初始化数据库管理器，创建数据库引擎和会话工厂。
    create_tables(self): 使用 SQLAlchemy 的 Base 元数据创建所有表。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.base import Base


class DatabaseManager:
    def __init__(self, db_uri="sqlite:///container_optimization.db"):
        # 自动添加sqlite:///前缀如果用户只提供了文件名
        if not db_uri.startswith("sqlite:///") and not db_uri.startswith("sqlite://"):
            if not db_uri.endswith(".db"):
                db_uri += ".db"
            db_uri = f"sqlite:///{db_uri}"

        self.engine = create_engine(db_uri)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """获取新会话"""
        return self.Session()


if __name__ == "__main__":
    db = DatabaseManager("sqlite:///container_optimization.db")
    db.create_tables()
