# 该程序绘制出训练集和测试集的准确率的柱状图，并且，用线段表示单步增长率的大小，比如time_steps=6，那么就是（准确率6-准确率1）/6
# time_steps=10，则是（准确率10-准去率1）/10

import numpy as np

hierarchical_acc = np.array([0.825305176, 0.916955566, 0.944860840, 0.963183594, 0.975524902, 0.977636719, 0.981408918, 0.984570313, 0.986389160, 0.987658691, 0.988830566])   # 分层注意力机制模型
hierarchical_acc = hierarchical_acc[::-1]
hierarchical_yerr = hierarchical_acc[:-1] - hierarchical_acc[1:]
hierarchical_yerr = np.insert(hierarchical_yerr, 10, 0) / 2

single_acc = np.array([0.824305176, 0.914904785, 0.934570313, 0.961682129, 0.969958496, 0.976330566, 0.980297852, 0.984216309, 0.983654785, 0.986460449, 0.987292480])   # dot-product注意力机制模型
single_acc = single_acc[::-1]
single_yerr = single_acc[:-1] - single_acc[1:]
single_yerr = np.insert(single_yerr, 10, 0) / 2

birgu_acc = np.array([0.82380836, 0.902526855, 0.932861328, 0.960949707, 0.968640137, 0.973901367, 0.978662109, 0.982434082, 0.984954889, 0.986254883, 0.987524414])   # 无注意力机制的模型
birgu_acc = birgu_acc[::-1]
birgu_yerr = birgu_acc[:-1] - birgu_acc[1:]
birgu_yerr = np.insert(birgu_yerr, 10, 0) / 2



# 获取横纵标
x_start = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33])

# 进行画图
import matplotlib.pyplot as plt

total_width, n = 1.8, 3
width = total_width / n

x_start = x_start - (total_width - width)/2


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

fig = plt.figure(1)
ax2 = fig.add_subplot(111)
ax2.bar(x_start, hierarchical_acc[::-1], width=width, label='multi-level attention', yerr=hierarchical_yerr[::-1])
ax2.bar(x_start + width, single_acc[::-1], width=width, label='single-level attention', yerr=single_yerr[::-1])
ax2.bar(x_start + 2 * width, birgu_acc[::-1], width=width, label='no attention', yerr=birgu_yerr[::-1])

ax2.set_ylim(0.8, 1.0)
ax2.set_xticks(x_start+3/2*width)
ax2.set_xticklabels(('1', '2', '3', '4', '5', '6','7', '8', '9', '10', '11'))
plt.xlabel('timestep ', font2)
plt.ylabel('Testing Accuracy', font2)
plt.legend(loc='upper left', prop=font2)
plt.title('Accuracy of Testing Set with Different Number of Timestep', font1)

plt.show()



