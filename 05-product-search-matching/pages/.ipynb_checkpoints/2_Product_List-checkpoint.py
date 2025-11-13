import streamlit as st
import requests

st.title("📋 2. 商品列表")

PRODUCT_LIST_URL = "http://localhost:8000/product/list"
headers = {'accept': 'application/json'}

if st.button("获取所有商品列表"):
    try:
        response = requests.get(PRODUCT_LIST_URL, headers=headers)
        
        st.subheader("API 响应")
        st.code(f"URL: {PRODUCT_LIST_URL}", language='http')
        st.metric(label="状态码", value=response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            # 显示为表格
            st.dataframe(data["data"]["products"]) 
        else:
            st.error(f"获取列表失败 (Status: {response.status_code})")
            st.code(response.text, language='json')

    except requests.exceptions.ConnectionError:
        st.error("无法连接到 FastAPI 服务。")
    except Exception as e:
        st.exception(e)