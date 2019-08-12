from matplotlib import pyplot as plt

x1 = [0, 0.1, 0.17, 1.19, 3.2, 4.47, 4.74, 5]
y1 = [0, 85, 87, 99, 100, 100, 100, 100]
plt.plot(x1, y1, color='r', marker='o', label='Our Mthod')

x2 = [0, 0.4, 0.7, 0.84, 1.34, 2.1, 2.2, 3.4, 3.67, 5.3]
y2 = [0, 11, 19, 22, 36, 40, 45, 57, 60, 62]
plt.plot(x2, y2, marker='v', label='LDM')

x3 = [0, 0.3, 0.4, 0.8, 1.56, 1.9, 2.7, 3.3, 3.8, 5]
y3 = [0, 21, 44, 46, 68, 73, 74, 77, 88, 90]
plt.plot(x3, y3, marker='^', label='NTHT')

x4 = [0, 0.21, 0.326, 0.6, 1.12, 1.64, 1.65, 2.3, 3.6, 4.1]
y4 = [0, 40, 61, 72, 74, 82, 88, 88, 92, 96]
plt.plot(x4, y4, marker='x', label='MPCM')

x4 = [0, 0.13, 0.646, 0.757, 1.012, 1.314, 1.615, 2.012, 3.143, 3.901]
y4 = [0, 69, 71, 88, 95, 99, 100, 100, 100, 100]
plt.plot(x4, y4, marker='+', label='TGRS')

x4 = [0, 0.123, 0.536, 0.91, 1.22, 1.634, 1.95, 3.022, 3.83, 4.31]
y4 = [0, 63, 80, 91, 91, 93, 92, 93, 96, 96]
plt.plot(x4, y4, marker='*', label='IPM')

x4 = [0, 0.1, 0.36, 0.567, 1.162, 1.74, 2.75, 3.12, 3.73, 4.2]
y4 = [0, 71, 88, 94, 93, 96, 99, 98, 100, 100]
plt.plot(x4, y4, marker='<', label='GST')




plt.xlabel('False alarm rate(Fa)')
plt.ylabel('Detection ratio(Pd)')

plt.legend()
plt.show()
