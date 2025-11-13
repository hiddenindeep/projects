import streamlit as st

st.set_page_config(
    page_title="🚀 FastAPI 商品管理 Demo",
    page_icon="🛍️",
)

st.title("🛍️ FastAPI 商品管理系统 Demo")
st.sidebar.success("请在左侧导航栏选择操作。")

st.markdown(
    """
    这是一个用于演示如何通过 **Streamlit** 界面来调用 **FastAPI** 商品管理 API 的应用。

    **功能列表:**
    - **服务健康检查**: 确保后端服务正常运行。
    - **商品列表**: 获取所有已创建的商品。
    - **创建商品**: 上传图片和标题来创建新商品。
    - **获取商品**: 根据 ID 查看单个商品详情。
    - **删除商品**: 根据 ID 删除商品。
    - **更新商品**: 修改商品的标题或图片。
    """
)