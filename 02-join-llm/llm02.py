import openai
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal,Optional
from datetime import datetime
from typing import List

client = openai.OpenAI(
    api_key="sk-b2fcae19cd1f4a7dbe605ce9fc8ef3be", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            #print("原始JSON:", arguments)  # 添加调试信息
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("验证错误:", str(e))  # 打印具体的验证错误
            print("原始响应:", response.choices[0].message)
            return None

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别,意图类型,实体标签"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather',
                    'match', 'map', 'website', 'news', 'message', 'contacts', 'translation',
                    'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone',
                    'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email',
                    'bus', 'story'
                    ] = Field(description="领域类别")
    intent: Literal[
                    'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
                    'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
                    'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
                    'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'
                    ] = Field(description="意图类型")
    Src: Optional[str] = Field(default=None, description="出发地")
    Des: Optional[str] = Field(default=None, description="目的地")
    date: Optional[str | datetime] = Field(default=None, description="日期或时间")
    person: Optional[str] = Field(default=None, description="人名")
    contact: Optional[str] = Field(default=None, description="联系人")
    organization: Optional[str] = Field(default=None, description="机构名")
    company: Optional[str] = Field(default=None, description="公司名")

    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            try:
                # 尝试解析不同格式的日期
                for fmt in ('%Y年%m月%d日', '%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S'):
                    try:
                        return datetime.strptime(v, fmt)
                    except ValueError:
                        continue
                # 如果都不匹配，返回原始字符串
                print(f"无法解析日期格式: {v}")  # 添加调试信息
                return v
            except Exception as e:
                print(f"日期解析错误: {e}")
                return v
        return v

def llm_prompt_tools(user_prompt):
    return ExtractionAgent(model_name = "qwen-plus").call(user_prompt, IntentDomainNerTask)

result = llm_prompt_tools("你能帮我查一下2024年1月1日从北京南站到上海的汽车票吗？")
print(result)