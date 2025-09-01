#定义一个Book类
class Book:
    def __init__(self):
        self.name = '爵迹'
        self.price = 39
        self.autor = '郭敬明'
        ##私有属性：以__开始，只能在本类调用
        self.__color = '黄色'

    def you(self):
        print('努力学习%s图书' % self.name)

    def info(self):
        print('图书名称：%s，价格：%d，作者：%s,颜色：%s' % (self.name, self.price, self.autor, self.__color))

b = Book()
b.you()
b.info()
# print(b.__color)

class Animal(object):
    def run(self):
        print('Animal run')

class Dog(Animal):
    def run(self):
        print('Dog run')

class Cat(Animal):
    def run(self):
        print('Cat run')

def run_two(a):
    a.run()
    a.run()

dog = Dog()
cat = Cat()

run_two(dog)
run_two(cat)

try:
    print(1/0)
except ZeroDivisionError:
    print('除数不能为0')
finally:
    print('程序结束')

import time
print(time.localtime())

#查看模块
#help('modules')