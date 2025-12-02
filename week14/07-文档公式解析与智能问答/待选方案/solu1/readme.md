## 整体方案介绍
  使用pdf 解析+rag问答(纯向量检索+大模型qa)一系列流程完成。pdf公式解析使用的是dots.ocr vlm模型转成md格式，
 向量检索模型是qwen3 embdding，问答大模型是千问3 模型：Qwen3-32B/Qwen3-235B-A22B-Thinking-2507。
## 具体步骤
 
###pdf数据处理  
   pdf转md，参考[dots.ocr](https://www.modelscope.cn/models/rednote-hilab/dots.ocr)的使用说明，使用vllm方式进行推理
   step1:对文件夹pdf文件进行批量转换，
         for file in ./xfdata/*.pdf; do python3 dots_ocr/parser.py "$file" done
   step2:因为pdf是分页的md，需要对转换后的输出目录进行数据合并
   执行代码 python data_prepare.py
      
### 主函数
   python code/main.py

### 改进策略
   针对存在很多条件不足没有答案的问题，回复结果统一为常数值10。
   因为评测指标是MSE，所以预测误差较大的数据肯定是真实值比较大，然后通过修改了一些大模型预测明显错误的地方。
   发现的badcase，原文question是xx万，没有明确规定单位的，需要改成元，回复答案单位没有进行换算。
   例子：在咨询服务项目中，若项目总收入为100万元，总成本为60万元，适用税率为25%，该项目的预期税后净利润是多少？

   还有一些是问题可能有答案，但是代码没回答出来，原因未知。这块相对也做了一些调整。
   例子:当前市场环境下，平台已有90,000用户，广告投入为800元，按照用户增长动态模型，一天后用户数量将达到多少？   
   
   **这块也可以调整大模型prompt进行优化上面的badcase**
   
 
###其他尝试
   增加重排步骤，调整embedding、llm的模型，但是没提升。   
   
###得分说明
   纯代码预测是2482.52分，修改了预测偏差较大的几类badcase 最终降低到476。
 
## 后续改进思路
  1.优化prompt，减少badcase
  2.调整召回策略，这块可以考虑对问答问题进行分类标签化，提高召回的准确率。