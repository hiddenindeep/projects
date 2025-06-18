#定义一个Book类
class Book:
    def __init__(self):
        self.name = '爵迹'
        self.price = 39
        self.autor = '郭敬明'

    def you(self):
        print('努力学习%s图书' % self.name)

    def info(self):
        print('图书名称：%s，价格：%d，作者：%s' % (self.name, self.price, self.autor))

class BookZi(Book):
    pass

book1 = Book()
book1.you()
book1.info()

book2 = BookZi()
book2.info()