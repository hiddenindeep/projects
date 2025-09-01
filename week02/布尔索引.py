import jieba
from typing import Dict,List,Set

#布尔索引-文本搜索类
class SercherScan():
    def __init__(self,title_file):
        #读取文件，将文件内容存入列表
        with open(title_file,'r',encoding='utf-8') as f:
            titiles = f.read()
        self.titles = list(set(titiles.split('\n')))
    
    #接收查询、转换查询、执行搜索、并打印结果
    def search(self,query):
        query_new = self.conv_query(query)
        print("######query_new######",query_new)
        for title in self.titles:
            if eval(query_new):
                print(title,query)
    
    #判断title是否包含了words中所有单词
    def word_match(self,words,title):
        ifmatch = True
        for word in ' '.join(jieba.cut(words)).split():
            if word != ' ' and word not in title:
                ifmatch = False
        return ifmatch
    
    #在title中高亮显示匹配到的关键词
    def highlighter(self,doc,word):
        for part in list(jieba.cut(word)):
            if part not in ('(',')','and','AND','or','OR','not','NOT',' '):
                doc = str(doc).replace(part,'<span style="color:red">{}</span>'.format(part))
        return doc
        
    #将查询文本转为可执行的python代码字符串
    def conv_query(self,query):
        query_new_parts = []
        for part in list(jieba.cut(query)):
            if part == '(' or part == ')':
                query_new_parts.append(part)
            elif part in ('and','AND','or','OR','not','NOT',' '):
                query_new_parts.append(part.lower())
            else:
                query_new_parts.append("self.word_match('{}',title)".format(part))
        query_new = ''.join(query_new_parts)
        return query_new

#倒排索引-文本搜索类
class SercherIIndex():
    def __init__(self,docs_file):
        #文档列表
        self.doc_list = []
        #索引字典{word : set(index of word)}
        self.index:Dict[str,Set[int]] = dict()

        #文档id
        self.doc_id = 0

        #读取文档
        with open(docs_file,'r') as f:
            docs_data = f.read()
        #将每个文档添加到文档列表中，并对文档中每个词建立索引字典dict{word:list[index of word]}
        for doc in docs_data.split('\n'):
            self.add_doc(doc)
    
    def add_doc(self,doc):
        #文档列表添加该文档
        self.doc_list.append(doc)

        #构建和更新文档中各单词对应的Postion集合
        for word in list(jieba.cut(doc)):
            if word in self.index:
                self.index[word].add(self.doc_id)
            else:
                self.index[word] = set([self.doc_id])
        self.doc_id += 1
        return self.doc_id - 1
    
    def word_match(self,words):
        #从倒排索引中找到包含word的文档id集合
        result = None
        for word in list(jieba.cut(words)):
            if result is None:
                result = self.index.get(word,set())
            else:
                result &= self.index.get(word,set())
        if result is None:
            result = set()
        return result
    
    def conv_query(self,query):
        '''
        args:待转换的原始查询字符串
        return:转换完成可通过eval执行返回id集合的代码段字符串
        eg: "苹果 and (华为 or 小米)" -> self.word_match('苹果')&(self.word_match('华为')|self.word_match('小米'))
        '''
        query_new_parts = []
        all_parts = list(jieba.cut(query))

        idx = 0
        while idx < len(all_parts):
            if all_parts[idx] == '(' or all_parts[idx] == ')':
                query_new_parts.append(all_parts[idx])
            elif all_parts[idx] == ' ':
                query_new_parts.append(' ')
            elif all_parts[idx] in ('and','and'):
                query_new_parts.append('&')
            elif all_parts[idx] in ('or','OR'):
                query_new_parts.append('|')
            elif all_parts[idx] in ('not','NOT'):
                query_new_parts.append('-')
            else:
                query_new_parts.append("self.word_match('{}')".format(all_parts[idx]))

            idx += 1
        query_new = ''.join(query_new_parts)
        return query_new
    
    def search(self,query):
        new_query = self.conv_query(query)
        print(new_query,eval(new_query))

        result = []
        for did in eval(new_query):
            result.append(self.doc_list[did])
        return result


# searcher = SercherScan('./week02/爬虫-新闻标题.txt')

# query = '苹果 and (芯片 or 高通)'
# result = searcher.search(query)
# query = 'iPhone and 摄像头'
# result = searcher.search(query)

searcher = SercherIIndex('./week02/爬虫-新闻标题.txt')
# print(searcher.doc_list)
# print(searcher.index)
query = '台湾 and (台独 or 中国)'
for doc in searcher.search(query):
    print(doc)

