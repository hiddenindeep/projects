from dashscope import MultiModalConversation
import dashscope
from PIL import Image

dashscope.api_key = "sk-cbf9e44f6f164d2b9d4b9bbf110bbd6c"

image_path = "./五大过程十大知识域.png"

response = MultiModalConversation.call(
    model="qwen-vl-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "image": f"file://{image_path}"
                },
                {
                    "text": "提取图中的文字转换为文本"
                }
            ]
        }
    ],
)

# 获取模型的回答
if response.status_code == 200:
    answer = response.output.choices[0].message.content[0]["text"]
    print("模型回答：", answer)
else:
    print("请求失败：", response.message)
