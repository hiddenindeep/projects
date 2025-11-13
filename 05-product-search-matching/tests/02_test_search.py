import requests
import json

url = "http://127.0.0.1:8000/product"
files = {
    'image': ('pokemon.jpeg', open('../pokemon.jpeg', 'rb'), 'image/jpeg')
}
data = {
    'title': '正常皮卡丘'
}
response = requests.post(url, files=files, data=data)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


url = "http://127.0.0.1:8000/product/search"
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "search_type": "text2text",
    "query_text": "机器学习",
    "top_k": 10
}

print("\n####")
print(url)
response = requests.post(url, headers=headers, data=json.dumps(data))
print(f"Status Code: {response.status_code}")
print(response.text)



url = "http://127.0.0.1:8000/product/search"
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "search_type": "text2image",
    "query_text": "宠物小精灵",
    "top_k": 10
}

print("\n####")
print(url)
response = requests.post(url, headers=headers, data=json.dumps(data))
print(f"Status Code: {response.status_code}")
print(response.text)