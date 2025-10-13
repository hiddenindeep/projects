# python自带库
import time
import traceback

# 第三方库
from fastapi import FastAPI

# 自己写的模块
from data_schema import LlmResponse
from data_schema import LlmRequest
from llm01 import llm_prompt
from llm02 import llm_prompt_tools

app = FastAPI()

@app.post("/v1/llm/prompt")
def gpt_classify(req: LlmRequest) -> LlmResponse:
    """
    利用大语言模型进行文本抽取领域类别,意图类型,实体标签

    :param req: 请求体
    """
    start_time = time.time()
    response = LlmResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        result="",
        run_time=0,
        error_msg=""
    )

    try:
        response.result = llm_prompt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.run_time = round(time.time() - start_time, 3)
    return response

@app.post("/v1/llm/tools")
def gpt_classify(req: LlmRequest) -> LlmResponse:
    """
    利用大语言模型进行文本抽取领域类别,意图类型,实体标签

    :param req: 请求体
    """
    start_time = time.time()
    response = LlmResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        result="",
        run_time=0,
        error_msg=""
    )

    try:
        response.result = llm_prompt_tools(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.run_time = round(time.time() - start_time, 3)
    return response
