import pandas as pd
import jieba

dataset = pd.read_csv("./week01/dataset.csv", sep="\t", header=None)
#print(dataset.head(5))

inverted_index = {} # word:list[index of word]

#保存每个词出现的索引集合
for i,row in dataset.iterrows():
    #文本
    text = row[0]
    #分词
    words = jieba.cut(text)
    for word in words:
        #如果该词在字典中不存在
        if word not in inverted_index:
            #在字典中添加该词
            inverted_index[word] = set() #set() 是无序不重复的元素集
        inverted_index[word].add(i)

def search(query):
    query_words = jieba.cut(query)
    result = None
    for word in query_words:
        if word in inverted_index:
            if result is None:
                result = inverted_index[word].copy()
            else:
                #result &= inverted_index[word] #交集:同时出现在两个集合中的元素
                result |= inverted_index[word] #并集:出现在任意一个集合中的元素
        else:
            return []
    
    if result is None:
        return []
    else:
        return list(result)
    
query = "长沙湘潭汽车票"
result_indexs = search(query)
print(f"搜索'{query}'的结果是：")
print(dataset.iloc[result_indexs].head(5))
        
    




