import matplotlib.pyplot as plt

x = [0, 0.01, 0.02, 0.03, 0.346, 0.477, 0.512, 0.6314, 0.7665, 0.912]
y = [0, 0.18, 0.26, 0.44, 0.79, 0.86, 0.91, 0.93, 0.88, 0.91]

plt.plot(x, y)

plt.ylabel('Accuracy')
plt.xlabel('Parameter-lambda')

plt.legend()
plt.show()
