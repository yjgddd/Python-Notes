import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#通过rcParams设置全局纵横轴字体大小
mpl.rcParams['xtick.labelsize']=24
mpl.rcParams['ytick.labelsize']=24

np.random.seed(42)

#x轴的采样点
x=np.linspace(0,5,100)

#通过下面曲线加上噪声生成数据
'''numpy.random.normal(loc=0.0, scale=1.0, size=None)
   正态分布的拟合，参数的意义为：
   loc:float 此概率分布的均值（对应着整个分布的中心centre)
   scale:float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小越高瘦）
   size：int or tuple of ints 输出的shape，默认为None，只输出一个值
   np.random.randn(size)是标准正态分布（μ=0,σ=1），
   对应于np.random.normal(loc=0, scale=1, size)。
'''
y=2*np.sin(x)+0.3*x**2
y_data=y+np.random.normal(scale=0.3,size=100)

#figure()指定图表名称
plt.figure('data')

#'.'标明画散点图，每个散点图的形状是个圆
plt.plot(x,y_data,'.')

#画模型的图，plot函数默认画连线图
plt.figure('model')
plt.plot(x,y)

#两个图画在一起
plt.figure('data&model')

# 通过'k'指定线的颜色，lw指定线的宽度
# 第三个参数除了颜色也可以指定线形，比如'r--'表示红色虚线
# 更多属性可以参考官网：http://matplotlib.org/api/pyplot_api.html
plt.plot(x, y, 'k', lw=3)

#scatter可以更容易的生成散点图
plt.scatter(x,y_data)

#将当前figure的图保存到文件result.png
plt.savefig('result1.png')

#一定要加上这条语句才能让画好的图显示在屏幕上
plt.show()


