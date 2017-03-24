# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

Y = [5.25, 2.875, 1.25, 1.5, 1.25, 1]
X = [50, 100, 200, 400, 800, 1400]

plt.plot(X,Y)
plt.xlabel('Training Size')
plt.ylabel('Test Error')
plt.title('SVM Test error against training size')
