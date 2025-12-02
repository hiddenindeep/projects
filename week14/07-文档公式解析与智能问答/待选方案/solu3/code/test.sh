#!/bin/bash

# 设置Python解释器路径 (评审方可能会修改这里)
# PYTHON_EXECUTABLE=python3
PYTHON_EXECUTABLE=python

# 获取脚本所在的目录，以便我们能正确地找到其他脚本
BASEDIR=$(dirname "$0")

echo "========================================================"
echo "INFO: 开始执行阶段一：LLM推理..."
echo "========================================================"

# 运行第一个脚本
$PYTHON_EXECUTABLE "$BASEDIR/1_run_inference.py"

# 检查第一个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "ERROR: 阶段一 (1_run_inference.py) 执行失败，流程终止。"
    exit 1
fi

echo ""
echo "========================================================"
echo "INFO: 阶段一完成。开始执行阶段二：后处理并生成提交文件..."
echo "========================================================"

# 运行第二个脚本
$PYTHON_EXECUTABLE "$BASEDIR/2_generate_submission.py"

# 检查第二个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "ERROR: 阶段二 (2_generate_submission.py) 执行失败，流程终止。"
    exit 1
fi

echo ""
echo "========================================================"
echo "INFO: 流程执行完毕！"
echo "INFO: 最终提交文件已生成在 'prediction_result/result'。"
echo "========================================================"