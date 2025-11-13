import streamlit as st
import requests

st.title("🗑️ 5. 删除商品")

base_url = "http://127.0.0.1:8000/product/"

product_id_to_delete = st.number_input("输入要删除的商品 ID", min_value=1, step=1)

if st.button("执行删除"):
    if product_id_to_delete:
        url = f"{base_url}{product_id_to_delete}"
        st.warning(f"即将删除 ID 为 **{product_id_to_delete}** 的商品。")
        
        if st.checkbox("我确认要删除此商品"):
            try:
                response = requests.delete(url)
                
                st.subheader("API 响应")
                st.code(f"URL: {url}", language='http')
                st.metric(label="状态码", value=response.status_code)
                
                if response.status_code == 200:
                    st.success(f"🗑️ 商品 ID **{product_id_to_delete}** 已成功删除。")
                    st.json(response.json())
                elif response.status_code == 404:
                    st.warning("商品不存在，无法删除 (Status: 404 Not Found)。")
                else:
                    st.error(f"删除失败 (Status: {response.status_code})")
                    st.code(response.text, language='json')

            except requests.exceptions.ConnectionError:
                st.error("无法连接到 FastAPI 服务。")
            except Exception as e:
                st.exception(e)
        else:
            st.info("请勾选确认框以执行删除操作。")
    else:
        st.warning("请输入商品 ID。")