import numpy as np

# Define Q, c, A, and b
Q = np.array([[6, 2, 1],
              [2, 5, 2],
              [1, 2, 4]], dtype=float)

c = np.array([-1, -1, -1], dtype=float)

A = np.array([[1, 0, 1],
              [0, 1, 1]], dtype=float)

b = np.array([3, 0], dtype=float)

# Build the KKT matrix
KKT_matrix = np.block([
    [Q, -A.T],
    [A, np.zeros((2, 2))]
])

# Right-hand side vector
rhs = np.concatenate([-c, b])

# Solve the KKT system
solution = np.linalg.solve(KKT_matrix, rhs)

# Extract x and lambda
x = solution[:3]
lambdas = solution[3:]

# Output results
print("Optimal x:", x)
print("Lagrange multipliers:", lambdas)

# Evaluate the objective function at the solution
f_opt = 3*x[0]**2 + 2.5*x[1]**2 + 2*x[2]**2 + 2*x[0]*x[1] + x[0]*x[2] + 2*x[1]*x[2] - x[0] - x[1] - x[2]
print("Optimal objective value:", f_opt)

# Eigen values of Q
eigenvalues = np.linalg.eigvals(Q)
print("Eigenvalues of Q:", eigenvalues)

# Rank of A
rank_A = np.linalg.matrix_rank(A)
print("Rank of A:", rank_A)