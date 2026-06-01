# Q1 a
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
  return x**3 - 3*x + y**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the contour plot
contours = plt.contour(X, Y, Z, levels=32, cmap='viridis')
plt.clabel(contours, inline=1, fontsize=8)

# Highlight the points (-1,0) and (1,0)
plt.scatter(-1, 0, color='red', marker='x', label="Saddle (-1,0)")
plt.scatter(1, 0, color='green', marker='o', label="Local Minima (1,0)")

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper right") # Show the legend

# Display the plot
plt.show()

