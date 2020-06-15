import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/nihaopeng/工作/实习/标定板照片/oppo_1.jpg')
plt.imshow(img)
plt.show()

print(img.shape)


R_t = [[0.143032449354255, -0.018807235045861, 0.006773818432406, -3.128035698574597e+02],
     [0.232338102429488, -0.022053121882626, -0.008711863103865, -2.889485151005842e+02],
     [0.297108851999945, 0.013893036145222, 0.015872215528216, 8.938263293777457e+02],]

K = [[3.876284499785856e+03, 0, 0],
     [0, 3.472544135974627e+03, 0],
     [2.085643205998702e+03, 1.720325784869812e+03, 1]]

for i in range(img.shape[2]):
     for x in range(img.shape[0]):
          for y in range(img.shape[1]):

