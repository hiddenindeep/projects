for i in range(1,10):
    for j in range(1,i+1):
        print(i,"*",j,"=",i*j,end="  ")
    print()

#计算阶乘
def jc(i):
    # 如果i等于1，返回1
    if i == 1:
        return 1
    # 否则，返回i乘以jc(i-1)
    else:
        return i*jc(i-1)

print(jc(5))
