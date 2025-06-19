#定义一个列表，并按降序排序
li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
li.sort(reverse=True)
print(li)

#判断是否为偶数
def isOs(num1):
    if num1 % 2 == 0:
        return True
    else:      
        return False
print(isOs(3))

#判断是否为偶数
print((lambda x: x % 2 == 0)(3))

#使用匿名函数对字典中的列表进行排序
#源数据：[{'name': '张三', 'age': 18}, {'name': '李四', 'age': 20}, {'name': '王五', 'age': 17}]
#排序规则：按照年龄升序排序
li = [{'name': '张三', 'age': 18}, {'name': '李四', 'age': 20}, {'name': '王五', 'age': 17}]
li.sort(key=lambda x: x['age'], reverse=False)
print(li)


#使用正则表达式匹配全部字符串输出
import re
s = 'hello 22world'
print(re.findall(r'\w+', s))
print(re.findall(r'\d+', s))
print(re.findall(r'[a-zA-Z]+', s))


#源数据:'hello 7709 badou' 输出'hello 7709 789 badou'
s1 = 'hello 7709 badou'
ss = re.sub(r'(\d+)(\s+)', r'\1 789\2',s1)
print(ss)