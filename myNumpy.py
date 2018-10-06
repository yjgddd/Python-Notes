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

