import numpy as np
import math

def get_bmr_rate(weight: float, height: float):
    """
    在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理
功能所需能量的重要指标。它广泛应用于能量需求评估、体重管理、临床营养支持等多个场
景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，
构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所
需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算。
    :param weight: 个体体重，单位为千克（kg）；
    :param height: 个体身高，单位为厘米（cm）。
    :return: 基础代谢率（BMR）
    """
    return round(10 * weight + 6.25 * height - 100, 2)

