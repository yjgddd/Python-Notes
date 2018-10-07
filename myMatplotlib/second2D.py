#柱状图、饼状图
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#通过rcParams设置全局纵横轴字体大小
mpl.rcParams['axes.titlesize']=20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

#包含了狗、猫和猎豹的最高奔跑速度，还有对应的可视化颜色。这是一个字典
speed_map={
    'dog':(48,'#7199cf'),
    'cat': (45, '#4fc4aa'),
    'cheetah': (120, '#e1a7a2')
}

'''
  fig=plt.figure() Figure实例，可以添加Axes实例
  ax=fig.add_subplot(121) 是在Figure上添加Axes的常用方法，它返回Axes实例
  参数一，子图总行数
  参数二，子图总列数
  参数三，子图位置
'''
#整体图的标题是'Bar chart & Pie chart'
fig=plt.figure('Bar chart & Pie chart')

# 在整张图上加入一个子图，121的意思是在一个1行2列的子图中的第一张
ax = fig.add_subplot(121)
ax.set_title('Running speed - bar chart')

'''
生成x轴每个元素的位置 arrange函数用来生成等差数组
arange([start,] stop[, step,], dtype=None) 
四个参数，其中start，step，dtype（可以省略），分别是起始点、步长和返回类型
'''
xticks=np.arange(3) #起始点0，结束点3，步长1，返回类型array，一维

#定义柱状图每个柱的宽度
bar_width=0.5

#动物名称
animals=speed_map.keys()

#奔跑速度
speeds=[x[0] for x in speed_map.values()]

#对应颜色
colors=[x[1] for x in speed_map.values()]

#画柱状图，横轴是动物标签的位置，纵轴是速度，定义柱的宽度，同时设置柱的边缘为透明
bars=ax.bar(xticks,speeds,width=bar_width,edgecolor='none')

#设置y轴的标题
ax.set_ylabel('Speed(km/h)')

#x轴每个标签的位置，设置为每个柱的中间
ax.set_xticks(xticks)

#设置每个标签的名字
ax.set_xticklabels(animals)

#设置x轴的范围
ax.set_xlim([bar_width/2-0.5,3-bar_width/2])

#设置y轴的范围
ax.set_ylim(0,125)

'''
给每个bar分配指定的颜色
zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。'''
for bar,color in zip(bars,colors):
    bar.setcolor(color)

#在122位置加入新的图
ax=fig.add_subplot(122)
ax.set_title('Running speed - pie chart')

'''
生成同时包含名称和速度的标签 
format()是格式化函数，Python2.6 开始，新增了一种格式化字符串的函数 str.format()，它增强了字符串格式化的功能。
基本语法是通过 {} 和 : 来代替以前的 % 
'''
labels=['{}\n{} km/h'.format(animal,speed) for animal, speed in zip(animals, speeds)]

#画饼状图，并指定标签和对应颜色
ax.pie(speeds, labels=labels, colors=colors)
plt.show()
