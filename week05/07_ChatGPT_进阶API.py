import openai
import json

client = openai.OpenAI(
    api_key="sk-WkCbMVOViwqUVVdD97E9E88612A14071A40213E24c2989Ab",
    base_url="https://openkey.cloud/v1"
)



response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content":
"""
请帮我进行文本分类，判断下面的文本是正向情感还是负面情感。请直接输出类别，不要有其他输出，可选类别：正/负

我今天很开心。
"""}],
    stream=False,
    logprobs = True,
    top_logprobs = 5
)
print(response)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": "李华和小王是不是认识？"},
    ],
    functions=[
        {
            "name": "get_connection",
            "description": "判断用户1和用户2 是否为朋友关系",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id1": {
                        "type": "string",
                        "description": "用户ID 1"
                    },
                    "user_id2": {
                        "type": "string",
                        "description": "用户ID 2"
                    },
                },
                "required": ["user_id1", "user_id2"]
            }
        }
    ]
)

print(response.model_dump_json(indent=4))

def get_connection(user_id1, user_id2):
    print(f"现在开始判断 {user_id1}  {user_id2} 之间的关系")


if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    function_args_str = response.choices[0].message.function_call.arguments

    # 将 JSON 字符串解析为 Python 字典
    function_args = json.loads(function_args_str)

    print(f"\n模型请求调用函数: {function_name}")
    print(f"参数: {function_args}\n")

    # --- 推荐的、更安全的方法 ---
    # 使用函数映射来调用本地函数，这比 eval 更安全
    function_map = {
        "get_connection": get_connection
    }

    if function_name in function_map:
        local_function = function_map[function_name]
        result = local_function(**function_args)
        print(result)
