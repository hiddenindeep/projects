import time
import os
os.environ["OPENAI_API_KEY"] = "sk-78cc4e9ac8f44efdb207b7232e1ae6d8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from agents import Agent, Runner, SQLiteSession
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 多轮对话 上下文，sqlite mysql
# 在相同agent  或 不同agent 之间共享历史对话
session = SQLiteSession("conversation_123")

async def main():
    agent = Agent(
        name="Assistant",
        model="qwen-max",
        instructions="Reply very concisely.", # 系统提示词 system messgae
    )
    
    result = await Runner.run(
        agent,
        "我叫王五，请给我讲笑话", # 用户的第一次输入 user message
        session=session
    )
    print(result.final_output) # assistant message

    result = await Runner.run(
        agent,
        "帮我计算100 * 1000", # 用户第二次输入 user message
        session=session
    )
    print(result.final_output) # assistant message

    result = await Runner.run(
        agent,
        "我是谁？", # 用户第三次输入 user message
        session=session
    )
    print(result.final_output) # assistant message

if __name__ == "__main__":
    asyncio.run(main())