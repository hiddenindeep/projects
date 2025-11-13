import requests

url = "http://localhost:8000/health"
headers = {
    'accept': 'application/json'
}
response = requests.get(url, headers=headers)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


url = "http://localhost:8000/product/list"
headers = {
    'accept': 'application/json'
}
response = requests.get(url, headers=headers)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


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


url = "http://127.0.0.1:8000/product/1"
response = requests.get(url)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


url = "http://127.0.0.1:8000/product/2"
response = requests.delete(url)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


url = "http://127.0.0.1:8000/product/3/title"
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {
    'title': '机器学习'
}
response = requests.patch(url, headers=headers, data=data)
print("\n####")
print(url)
print(f"Status Code: {response.status_code}")
print(response.text)


url = "http://127.0.0.1:8000/product/3/image"
with open('../pokemon.jpeg', 'rb') as f:
    files = {
        'image': ('pokemon.jpeg', f, 'image/jpeg')
    }
    print("\n####")
    print(url)
    response = requests.patch(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(response.text)