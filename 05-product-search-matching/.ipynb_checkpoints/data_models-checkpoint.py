from typing import List, Optional
from pydantic import BaseModel


class ProductImageResponse(BaseModel):
    """
    商品图片信息响应模型
    """
    id: int
    image_path: str
    image_bytes: Optional[bytes] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        orm_mode = True


class ProductDescriptionResponse(BaseModel):
    """
    商品标题描述信息响应模型
    """
    id: int
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        orm_mode = True


class ProductInfoResponse(BaseModel):
    """
    完整商品信息响应模型（包含图片和描述）
    """
    image: ProductImageResponse
    description: ProductDescriptionResponse

    class Config:
        orm_mode = True
