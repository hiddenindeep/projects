from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Product(Base):
    """
    商品模型，纪录商品基本信息，也商品在milvus的信息
    整合了商品图片和描述信息
    """
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, index=True)

    # 图片文件路径或URL
    image_path = Column(String(500), nullable=False)

    # 商品标题文本
    title = Column(Text, nullable=False)

    # 创建时间
    created_at = Column(DateTime, default=datetime.datetime.utcnow) # 默认值

    # 更新时间
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    milvus_primary_key = Column(Integer, nullable=True) # 外键
    
    # 为标题字段创建索引以提高搜索性能
    __table_args__ = (
        Index('idx_title', 'title'),
    )
    
    def __repr__(self):
        return f"<Product(id={self.id}, title='{self.title[:50]}...', image_path='{self.image_path}')>"


# 创建表的函数
def create_tables(engine):
    """
    创建所有表
    """
    Base.metadata.create_all(bind=engine)