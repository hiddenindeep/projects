# https://github.com/openai/openai-agents-python
# pip install -qU openai-agents
# pip install "openai-agents[viz]"

import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-78cc4e9ac8f44efdb207b7232e1ae6d8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


agent = Agent(
    model="qwen-max", # 模型代号
    name="Assistant", # 给agent的取得名字（推荐英文，写的有意义）
    instructions="You are a helpful assistant" # 对话中的 开头 system message
)

result = Runner.run_sync(agent, "帮我写一个对联。") # 同步运行，输入 user messgae
print(result.final_output)


# python 00_openai-agent环境.py