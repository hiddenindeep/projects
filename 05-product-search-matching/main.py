from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel
import time
import base64
import uuid
import os
from PIL import Image
from io import BytesIO

from sqlalchemy.sql.roles import ReturnsRowsRole

# 导入数据模型
from orm_models import Product, create_tables
from data_models import BasicResponse, SearchRequest

# 导入向量数据库
import vector_db

# 数据库配置
SQLALCHEMY_DATABASE_URL = "sqlite:///./product.db" # mysql
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建数据库表
create_tables(engine)

# 创建图片保存目录
IMAGE_DIR = "./product_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 存储服务启动时间
start_time = time.time()

app = FastAPI()

# 获取数据库会话的依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """
    后端健康检查接口
    返回服务状态信息（程序的启动时间，是否连接到关系型数据库，是否连接到milvus）
    """
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "service": "Product Search and Matching Service",
        "uptime": f"{uptime:.2f} seconds",
        "milvus": vector_db.client.list_collections() 
    }


@app.post("/product", response_model=BasicResponse)
async def create_product(title: str = Form(...), image: UploadFile = File(...), db=Depends(get_db)):
    """
    新建商品（使用form表单提交）
    
    Args:
        title: 商品标题（form字段）
        image: 商品图片文件（文件上传字段，支持jpg、png等格式）
        db: 数据库会话
    
    Returns:
        创建的商品信息
    """
    try:
        # 验证文件类型
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传有效的图片文件")
        
        # 获取文件扩展名
        file_extension = image.filename.split(".")[-1] if "." in image.filename else "jpg"
        
        # 生成唯一文件名，不会用上传的文件名
        unique_filename = str(uuid.uuid4()) + f".{file_extension}"
        image_path = os.path.join(IMAGE_DIR, unique_filename)
        
        # 保存图片到本地
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # milvus，插入商品到向量数据库
        insert_result, milvus_primary_key = vector_db.insert_product(image_path, title)

        # 关系型数据库：创建商品实例，使用相对路径存储
        db_product = Product(
            title=title,
            image_path=image_path,
            milvus_primary_key=milvus_primary_key
        )
        # 保存到数据库
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        
        return BasicResponse(
            status=200,
            message="商品创建成功",
            data={
                "id": db_product.id,
                "created_at": db_product.created_at
            }
        )

    except Exception as e:
        # 撤销当前事务中的所有数据库操作，将数据库恢复到事务开始前的状态。
        db.rollback()
        raise HTTPException(status_code=400, detail=f"图片处理失败: {str(e)}")


@app.get("/product/list", response_model=BasicResponse)
async def list_products(db=Depends(get_db)):
    """
    获取所有商品列表
    
    Args:
        page_size: 查询页面的大小
        page_index: 具体查询的页面
        order: 排序逻辑（插入、更新）
        db: 数据库会话
    
    Returns:
        包含所有商品信息的列表
    """

    # TODO 需要分页，page size / page index

    # TODO 需要排序，按照插入时间 / 按照更新时间排序
    
    # 关系型数据库查询
    products_data = db.query(Product.id, Product.title, Product.image_path, Product.created_at, Product.updated_at, Product.milvus_primary_key).all()

    products = []
    for product in products_data:
        product_dict = {
            "id": product.id,
            "title": product.title,
            "image_path": product.image_path,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
            "milvus_primary_key": product.milvus_primary_key
        }
        products.append(product_dict)
    
    return BasicResponse(
        status=200,
        message="查询结果成功",
        data={
            "products": products,
            "total": len(products)
        }
    )


@app.get("/product/{product_id}", response_model=BasicResponse)
async def get_product(product_id: int, db=Depends(get_db)):
    """
    获取商品详情
    
    Args:
        product_id: 商品ID
        db: 数据库会话
    
    Returns:
        商品详细信息
    
    Raises:
        HTTPException: 当商品不存在时返回404错误
    """
    # 查询商品
    product = db.query(Product).filter(Product.id == product_id).first()
    
    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        product_dict = {
            "id": product.id,
            "title": product.title,
            "image_path": product.image_path,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
            "milvus_primary_key": product.milvus_primary_key
        }
        return BasicResponse(
            status=200,
            message="查询结果成功",
            data=product_dict
        )


@app.patch("/product/{product_id}/title", response_model=BasicResponse)
async def update_product_title(product_id: int, title: str = Form(None), db=Depends(get_db)):
    """
    更新商品信息
    
    Args:
        product_id: 商品ID
        title: 商品标题（form字段）
        db: 数据库会话
    
    Returns:
        更新后的商品信息
    
    Raises:
        HTTPException: 当商品不存在时返回404错误
    """
    # 查询商品
    product = db.query(Product).filter(Product.id == product_id).first()
    
    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        # 更新商品字段
        update_data = {"title": title}
        for field, value in update_data.items():
            setattr(product, field, value)
        
        # 先删除，后插入
        vector_db.delete_product([product.milvus_primary_key])
        
        # 插入更新后的商品
        insert_result, milvus_primary_key = vector_db.insert_product(product.image_path, product.title)
        product.milvus_primary_key = milvus_primary_key

        # 保存更新
        db.commit()
        db.refresh(product)

        return BasicResponse(
            status=200,
            message="商品更新成功",
            data=None
        )   


@app.patch("/product/{product_id}/image", response_model=BasicResponse)
async def update_product_image(product_id: int, image: UploadFile = File(None), db=Depends(get_db)):
    """
    更新商品信息
    
    Args:
        product_id: 商品ID
        image: 商品图片文件（文件上传字段，支持jpg、png等格式）
        db: 数据库会话
    
    Returns:
        更新后的商品信息
    
    Raises:
        HTTPException: 当商品不存在时返回404错误
    """
    # 查询商品
    product = db.query(Product).filter(Product.id == product_id).first()
    
    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        try:
            # 保存图片到本地
            with open(product.image_path, "wb") as f:
                content = await image.read()
                f.write(content)

            vector_db.delete_product([product.milvus_primary_key])
            
            # 插入更新后的商品
            insert_result, milvus_primary_key = vector_db.insert_product(product.image_path, product.title)
            product.milvus_primary_key = milvus_primary_key
            
            # 保存更新
            db.commit()
            db.refresh(product)

            return BasicResponse(
                status=200,
                message="商品更新成功",
                data=None
            )   
        except Exception as e:
            return BasicResponse(
                status=400,
                message=f"图片处理失败: {str(e)}",
                data=None
            )


@app.delete("/product/{product_id}", response_model=BasicResponse)
async def delete_product(product_id: int, db=Depends(get_db)):
    """
    删除商品
    
    Args:
        product_id: 商品ID
        db: 数据库会话
    
    Returns:
        删除成功消息
    
    Raises:
        HTTPException: 当商品不存在时返回404错误
    """
    # 查询商品
    product = db.query(Product).filter(Product.id == product_id).first()
    
    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )    
    else:
        # 删除商品
        db.delete(product)
        db.commit()
        
        # 删除向量数据库中的商品
        vector_db.delete_product([product.milvus_primary_key])
        
        try:
            os.path.remove(product.image_path) # 本地保存的图
        except Exception as e:
            pass

        return BasicResponse(
            status=200,
            message="商品删除成功",
            data=None
        )


@app.post("/product/search", response_model=BasicResponse)
async def search_products(search_request: SearchRequest, db=Depends(get_db)):
    """
    商品搜索接口
    支持四种搜索模式：
    1. text2text: 文本搜索文本
    2. text2image: 文本搜索图片
    3. image2text: 图片搜索文本
    4. image2image: 图片搜索图片
    
    Args:
        search_request: 搜索请求参数，包含搜索类型、查询内容和返回数量
        db: 数据库会话
    
    Returns:
        搜索结果列表，按相似度排序
    
    Raises:
        HTTPException: 当请求参数无效时返回400错误
    """
    # 验证搜索类型
    valid_search_types = ["text2text", "text2image", "image2text", "image2image"]
    if search_request.search_type not in valid_search_types:
        return BasicResponse(
            status=400,
            message=f"无效的搜索类型，请使用以下类型之一: {', '.join(valid_search_types)}",
            data=None
        )

    # 根据搜索类型验证参数
    if search_request.search_type == "text2text":
        if not search_request.query_text:
            return BasicResponse(
                status=400,
                message="文本搜索模式需要提供query_text参数",
                data=None
            )
    elif search_request.search_type == "text2image":
        if not search_request.query_text:
            return BasicResponse(
                status=400,
                message="文本搜索模式需要提供query_text参数",
                data=None
            )
    elif search_request.search_type == "image2text":
        if not search_request.query_image:
            return BasicResponse(
                status=400,
                message="图片搜索模式需要提供query_image参数",
                data={}
            )
    elif search_request.search_type == "image2image":
        if not search_request.query_image:
            return BasicResponse(
                status=400,
                message="图片搜索模式需要提供query_image参数",
                data=None
            )

    try:
        if search_request.search_type in ["text2text", "text2image"]:
            success, results = vector_db.search_product(title=search_request.query_text, top_k=search_request.top_k, task=search_request.search_type)
        elif search_request.search_type in ["image2text", "image2image"]:
            image = Image.open(BytesIO(base64.b64decode(search_request.query_image)))
            success, results = vector_db.search_product(image=image, top_k=search_request.top_k, task=search_request.search_type)
        
        if not success:
            return BasicResponse(
                status=400,
                message="搜索失败",
                data=None
            )

        top_product_ids = [item["primary_key"] for item in results[0]]
        top_product_distance = [item["distance"] for item in results[0]]

        # 查询数据库中的商品信息
        top_products = db.query(Product).filter(Product.milvus_primary_key.in_(top_product_ids)).all()
        
        search_results = []
        for product in top_products:
            product_dict = product.__dict__
            product_dict["distance"] = top_product_distance[top_product_ids.index(product.milvus_primary_key)]
            search_results.append({
                "id": product.id,
                "title": product.title,
                "image_path": product.image_path,
                "created_at": product.created_at,
                "updated_at": product.updated_at,
                "milvus_primary_key": product.milvus_primary_key,
                "distance": product_dict["distance"] # 相似度
            })

        # 按照distance 较大的 cosine 值表示更高的相似性。
        search_results.sort(key=lambda x: x["distance"], reverse=True)

        return BasicResponse(
            status=200,
            message="搜索成功",
            data=search_results
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索过程中发生错误: {str(e)}")


@app.post("/product/generate", response_model=BasicResponse)
async def generate_product_image():
    """
    文本生成商品图片接口
    
    Args:
        generate_request: 图片生成请求参数，包含文本提示、生成数量和尺寸
    
    Returns:
        生成的图片路径列表
    
    Raises:
        HTTPException: 当请求参数无效或生成失败时返回错误
    """
    # 验证参数
    if not generate_request.text_prompt:
        raise HTTPException(status_code=400, detail="生成图片需要提供text_prompt参数")
    
    if generate_request.num_images < 1 or generate_request.num_images > 10:
        raise HTTPException(status_code=400, detail="生成图片数量必须在1-10之间")
    
    # 验证图片尺寸
    valid_sizes = ["256x256", "512x512", "1024x1024"]
    if generate_request.image_size not in valid_sizes:
        raise HTTPException(
            status_code=400, 
            detail=f"无效的图片尺寸，请使用以下尺寸之一: {', '.join(valid_sizes)}"
        )
    
    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片生成过程中发生错误: {str(e)}")
