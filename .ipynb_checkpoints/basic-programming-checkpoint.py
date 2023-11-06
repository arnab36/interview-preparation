# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:21:18 2021

@author: 01927Z744

Basic programming for brother

"""
'''

marks = int(input())

if marks < 360:
    print("You ave failed")
elif marks >= 360 and marks < 420:
    print("You got 3rd Division")
elif marks >= 420 and marks < 480:
    print("You got 2nd Division")
elif marks >= 480 and marks < 600:
    print("You got 1st Division")
else:
    print("you got Star")
    
'''    

'''

num = int(input("Please enter the number : "))

if num % 5 == 0:
    print("Number is divisible by 5")
    print("Number is divisible by 5")    
    print("Number is divisible by 5")
    print("Number is divisible by 5")
else:
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
'''

'''

num = int(input("Please enter the number : "))

if num % 5 != 0:
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
    print("Number is not divisible by 5")
else:       
    print("Number is divisible by 5")
    print("Number is divisible by 5")    
    print("Number is divisible by 5")
    print("Number is divisible by 5")
'''

'''
num1 = int(input())

num2 = int(input())

if num1 > num2:
    result = num1 - num2
else:
    result = num2 - num1

print(result)

'''

'''
limit = int(input('Enter the range: '))
divisible = int(input("please enter the divisor : "))

for k in range(0,limit+1):
    if k % divisible == 0:
        print(k)
   
'''

'''
def add(x,y):
    z = x - y
    return z


a = 1.2
b = 5
c = add(b,a)
print(c)
'''


'''
def namta(x):
    for i in range(1,11):
        print(i * x)


# namta(9)
namta(8)

'''

'''

def calculateDivision(marks):
    if marks < 360:
        print("You ave failed")
    elif marks >= 360 and marks < 420:
        print("You got 3rd Division")
    elif marks >= 420 and marks < 480:
        print("You got 2nd Division")
    elif marks >= 480 and marks < 600:
        print("You got 1st Division")
    else:
        print("you got Star")



calculateDivision(344)

'''

'''
file1 = open("any_name.txt","w")
file1.write("anything")
file1.close()

'''

'''
path = "lots-of-file/"

for j in range(1,501):
    file1 = open(path+str(j)+"_a.txt","w")
    for i in range(1,11):
        file1.write(str(i)+"&"+str(j)+"\n")
    file1.close()

'''













