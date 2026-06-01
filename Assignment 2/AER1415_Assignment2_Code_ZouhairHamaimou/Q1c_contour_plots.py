# Q1 c
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
  return (y-x**2)**2

x = np.linspace(-10, 10, 100)
y = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the contour plot
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=1, fontsize=8)

# Highlight the points (-1,0) and (1,0)
plt.plot(x, x**2, color='red', label="$y=x^2$")

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper right") # Show the legend

# Display the plot
plt.show()
