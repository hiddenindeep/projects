import streamlit as st
import requests

st.title("🩺 1. 服务健康检查")

HEALTH_URL = "http://localhost:8000/health"
headers = {'accept': 'application/json'}

if st.button("运行健康检查"):
    try:
        response = requests.get(HEALTH_URL, headers=headers)
        
        st.subheader("API 响应")
        st.code(f"URL: {HEALTH_URL}", language='http')
        
        st.metric(label="状态码", value=response.status_code)
        
        # 尝试解析 JSON 响应
        try:
            st.json(response.json())
        except requests.exceptions.JSONDecodeError:
            st.code(response.text, language='json')
            
        if response.status_code == 200:
            st.success("服务运行正常 (Status: 200 OK)")
        else:
            st.error(f"服务异常 (Status: {response.status_code})")

    except requests.exceptions.ConnectionError:
        st.error("无法连接到 FastAPI 服务。请确认服务已在 http://localhost:8000 运行。")
    except Exception as e:
        st.exception(e)