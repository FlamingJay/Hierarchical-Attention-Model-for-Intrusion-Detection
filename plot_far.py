
# 对比的是虚警率，其中hierarchical是分层注意力机制的模型，single是只有dot-product的模型，bigru是无注意力机制的模型
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


hierarchical = [0.343711498, 0.125946636, 0.093793201, 0.059762914, 0.035808357, 0.028992422, 0.024547005, 0.017632529, 0.014940728, 0.014965424, 0.012347709]
single = [0.347711498, 0.153728983, 0.117204458, 0.060923598, 0.04598287, 0.033116557, 0.02679453, 0.01918834, 0.017904179, 0.017706615, 0.014002303]
bigru = [0.351404638, 0.187117189, 0.112759283, 0.063491922, 0.048057285, 0.041389522, 0.029313462, 0.021559101, 0.017121772, 0.014743165, 0.013088572]


# example data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
xnew = np.linspace(x.min(), x.max(), 11)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14
}
 
fig, ax = plt.subplots()
smooth_hierarchical = spline(x, hierarchical, xnew)
plt.plot(xnew[::], smooth_hierarchical * 100, marker='o',label='multi-level attention')

smooth_single = spline(x, single, xnew)
plt.plot(xnew, smooth_single * 100, marker='^', label='single-level attention')

smooth_bigru = spline(x, bigru, xnew)
plt.plot(xnew, smooth_bigru * 100, marker='p', label='no attention')

# smooth_test_far_bigru = spline(x, test_far_bigru, xnew)
# plt.plot(xnew, smooth_test_far_bigru[::-1] * 100, marker='d', label='BiGRU')

ax.set_xticks(xnew)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'))
plt.grid(linestyle='-.')
plt.xlabel('timestep', font2)
plt.ylabel('False Alarm Rate %', font2)
plt.legend(loc='upper right', prop=font2)
plt.title('False Alarm Rate of Testing Set with Different Number of Timestep', font1)

plt.show()