import numpy as np
a=[1,2,3,4]
b=np.array(a)  #array([1,2,3,4])
tpye(b) #<type 'numpy.ndarray'> 返回数据类型
b.shape #(4,) shape()返回一个包含数组维度的元组，它也可以用于调整数组大小。
b.argmax() #3 argmax()返回最大数的索引
b.max()  #4
b.mean() #2.5 返回平均值

c=[[1,2],[3,4]] #二维列表
d=np.array(c) #二维numpy数组
d.shape #(2,2)
d.size #4
#np.max：(a, axis=None, out=None, keepdims=False)
#求序列的最值
#最少接收一个参数
#axis：默认为列向（也即 axis=0），axis = 1 时为行方向的最值；
d.max(axis=0) #array([3,4])
d.max(axis=1) #array([2,4])
d.mean(axis=0) #array([2., 3.])
d.flatten() #array([1, 2, 3, 4]) 展开一个numpy数组为1维数组
np.ravel(c) #array([1, 2, 3, 4]) 展开一个可以解析的结构为1维数组

#3*3的浮点型二维数组，并且初始化所有元素值为1
e=np.ones((3,3),dtype=np.float)
print(e)
''' 
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]'''

# 创建一个一维数组，元素值是把3重复4次，array([3, 3, 3, 3])
f=np.repeat(3,4)
print(f) #[3 3 3 3]

g=np.zeros((2,2,3),dtype=np.uint8)
g.shape()# (2,2,3)
h=g.astype(np.float) #用另一种类型表示

l=np.arange(10) #类似range，array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(l) #[0 1 2 3 4 5 6 7 8 9]
m=np.linspace(0,6,5) #等差数列，0到6之间5个取值，array([ 0., 1.5, 3., 4.5, 6.])
print(m)#[0.  1.5 3.  4.5 6. ]

p=np.array(
    [[1,2,3,4],
     [5,6,7,8]]
)
np.save('p.npy',p) #保存到文件
q=np.load('p.npy')#从文件读取

#数学运算
a=np.abs(-1) #绝对值
b=np.sin(np.pi/2) #sin函数，1.0
c=np.arctanh(0.462118) #tanh逆函数，0.5000010715784053
d=np.exp(3)#e为底的指数函数，20.085536923187668
f=np.power(2,3)#2的三次方，8
g=np.dot([1,2],[3,4])#点积，1*3+2*4=11
h=np.sqrt(25) #开方，5
l=np.sum([1,2,3,4]) #求和，10
m=np.mean([4,5,6,7]) #求平均值，5.5
p=np.std([1,2,3,2,1,3,2,0]) #求标准差，0.9682458365518543

''' 
对于array，默认执行对位运算。涉及到多个array的对位运算需要array的维度一致,如果一个array的维度和
另一个array的子维度一致，则在没有对齐的维度上分别执行对位运算，这种机制叫做广播（broadcasting）
'''
a=np.array([
    [1,2,3],
    [4,5,6]
])
b=np.array([
    [1,2,3],
    [1,2,3]
])
'''维度一样的array对位运算，相加的结果是
array([[2,4,6],
[5,7,9]'''
a+b
'''相减的结果是
array([[0, 0, 0],
       [3, 3, 3]])'''
a-b
'''乘法的结果是
array([[ 1,  4,  9],
       [ 4, 10, 18]])'''
a*b
'''除法的结果是
array([[1. , 1. , 1. ],
       [4. , 2.5, 2. ]])'''
a/b
'''a平方的结果是
array([[ 1,  4,  9],
       [16, 25, 36]], dtype=int32)'''
a**2
'''a的b次方的结果是
array([[  1,   4,  27],
       [  4,  25, 216]], dtype=int32)'''
a**b

c=np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
d = np.array([2, 2, 2])
'''广播机制让计算的表达式保持简介，d和c的每一行分别进行运算
其结果是
array([[ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14]])'''
c+d
'''c*d的结果是
array([[ 2,  4,  6],
       [ 8, 10, 12],
       [14, 16, 18],
       [20, 22, 24]])'''
c*d
'''1和c的每一个元素分别进行运算
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])'''
c-1
