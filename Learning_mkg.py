def num(limit):
    for i in range(0,limit+1):
        if i%3==0 or i%5==0:
            print(i)

x = int(input('enter number:  '))
num(x)




