from vector_db import client
import os
import glob

client.delete(collection_name="product_new", filter="ids not in [1, 2, 3]")
print("删除product_new集合完成")

if os.path.exists("product.db"):
    os.remove("product.db")
print("删除product.db数据库完成")

for path in glob.glob("product_images/*"):
    os.remove(path)
print("删除product_images目录下所有文件完成")
