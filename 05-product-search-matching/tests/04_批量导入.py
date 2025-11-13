import glob
import pandas as pd
import requests
import json

train = pd.read_csv("/root/autodl-tmp/dataset/多模态图文检索/train.csv", sep="\t")

for row in train.iloc[10:200].iterrows():
    url = "http://127.0.0.1:8000/product"
    files = {
        'image': (row[1].path, open("/root/autodl-tmp/dataset/多模态图文检索/image/" + row[1].path, 'rb'), 'image/jpeg')
    }
    data = {
        'title': row[1].title
    }
    response = requests.post(url, files=files, data=data)
    print("\n####")
    print(url)
    print(response.text)
    print(f"Status Code: {response.status_code}")