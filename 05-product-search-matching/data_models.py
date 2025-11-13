from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime

class BasicResponse(BaseModel):
    """
    基础响应模型
    """
    status: int
    message: str
    data: Optional[Union[dict, list]] = None

class SearchRequest(BaseModel):
    """
    搜索请求模型
    """
    search_type: str = "text2text"
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = 10
