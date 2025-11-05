from dashscope import Generation
import dashscope  # ✅ 一定要先 import dashscope

# 🔑 设置 API Key（替换为你自己的）
dashscope.api_key = "sk-cbf9e44f6f164d2b9d4b9bbf110bbd6c"

# 测试调用
resp = Generation.call(
    model="qwen-turbo",
    prompt="你好，测试一下API是否可用。"
)

print(resp.output_text)
