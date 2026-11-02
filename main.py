from scipy.optimize import linprog

# Minimise: 2x + 3y
# Subject to: x + y = 10
#             0 <= x <= 8
#             0 <= y <= 8

result = linprog(
    c=[2, 3],
    A_eq=[[1, 1]],
    b_eq=[10],
    bounds=[(0, 8), (0, 8)],
    method="highs",
)

if result.success:
    print("Optimal allocation:", result.x)
    print("Minimum cost:", result.fun)
else:
    print("Optimisation failed:", result.message)