import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial import cKDTree

np.random.seed(42)
points = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=500)

x_params = norm.fit(points[:, 0])
y_params = norm.fit(points[:, 1])

n_samples = 100
sampled_points = np.column_stack((
    np.random.normal(loc=x_params[0], scale=x_params[1], size=n_samples),
    np.random.normal(loc=y_params[0], scale=y_params[1], size=n_samples)
))

kdtree = cKDTree(points)
_, indices = kdtree.query(sampled_points)
final_samples = points[indices]

plt.figure(figsize=(10, 8))

plt.scatter(points[:, 0], points[:, 1], c='blue', label='Original Points', alpha=0.5)

plt.scatter(final_samples[:, 0], final_samples[:, 1], c='red', label='Sampled Points', marker='x')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Points and Sampled Points')
plt.show()
