import numpy as np

a = np.ones((4,1))
print(a)
b = np.arange(4)
print(b)

c = a + b
print(c)

#100个员工考勤天数-1
days = np.random.randint(20,23, size=(100,))
print(days)
result = days - 1
print(result)

#上班时间不足8小时
times = np.random.randint(7,10,100)
print(times)
result = times < 8
print(result)