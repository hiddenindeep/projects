# 导入必要的库和模块
from transformers import Qwen2_5_VLForConditionalGeneration,AutoTokenizer,AutoProcessor
from qwen_vl_utils import process_vision_info

# 从本地路径加载预训练的Qwen2.5-VL模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("../models/qwen2_5_vl/")

# 从本地路径加载与模型匹配的处理器
processor = AutoProcessor.from_pretrained("../models/qwen2_5_vl/")

# 定义输入消息，包含图像和文本提示
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",  # 指定内容类型为图像
                "image": "./data/pingpong.jpeg",  # 图像文件路径
            },
            {"type": "text", "text": "Describe this image."},  # 文本提示
        ],
    }
]

# 使用聊天模板处理对话消息
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# 处理视觉信息（图像和视频）
image_inputs, video_inputs = process_vision_info(messages)
# 使用处理器处理文本、图像和视频输入，进行填充并转换为PyTorch张量
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# 将输入张量移动到CUDA设备上
inputs = inputs.to("cuda")

# 使用模型生成新的token，最大生成长度为128
generated_ids = model.generate(**inputs, max_new_tokens=128)
# 去除输入部分的token，只保留新生成的部分
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
# 解码生成的token为文本，跳过特殊token，保留空格
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
# 打印生成的文本
print(output_text)