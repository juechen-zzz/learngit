from matplotlib import pyplot as plt

x1 = [0, 0.1, 0.17, 0.19, 0.2, 0.47, 0.74, 3.3]
y1 = [0, 80, 89, 93, 98, 99, 100, 100]
plt.plot(x1, y1, color='r', marker='o', label='Our Mthod')

x2 = [0, 0, 0.2, 0.4, 0.94, 1.1, 2.2, 2.4, 3.1, 3.3]
y2 = [0, 7, 9, 21, 26, 41, 45, 47, 50, 52]
plt.plot(x2, y2, marker='v', label='LDM')

x3 = [0, 0.3, 0.4, 0.8, 1.1, 1.6, 2.2, 2.3, 2.8, 3]
y3 = [0, 11, 34, 36, 48, 53, 54, 67, 80, 94]
plt.plot(x3, y3, marker='^', label='NTHT')

x4 = [0, 0.3, 0.36, 0.7, 1.2, 1.4, 1.5, 2.2, 2.3, 3.1]
y4 = [0, 40, 71, 70, 75, 79, 80, 88, 90, 96]
plt.plot(x4, y4, marker='x', label='MPCM')

x4 = [0, 0.33, 0.46, 0.57, 1.12, 1.14, 1.65, 2.12, 2.43, 3.01]
y4 = [0, 70, 81, 90, 95, 99, 100, 100, 100, 100]
plt.plot(x4, y4, marker='+', label='TGRS')

x4 = [0, 0.23, 0.336, 0.71, 1.02, 1.34, 1.95, 2.12, 2.83, 3.31]
y4 = [0, 60, 81, 90, 91, 92, 92, 93, 95, 96]
plt.plot(x4, y4, marker='*', label='IPM')

x4 = [0, 0.1, 0.536, 0.67, 1.62, 1.74, 1.75, 2.12, 2.73, 3.2]
y4 = [0, 70, 91, 90, 95, 97, 99, 98, 100, 100]
plt.plot(x4, y4, marker='<', label='GST')




plt.xlabel('False alarm rate(Fa)')
plt.ylabel('Detection ratio(Pd)')

plt.legend()
plt.show()
