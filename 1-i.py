# minimize: 60x0 + 40x1 + 50x2
# Subject to: -20x0 - 10x1 - 10x2 <= -350
#             -10x0 - 10x1 - 20x2 <= -400

from scipy.optimize import linprog

maxim = [60, 40, 50]
coef = [[-20, -10, -10], [-10, -10, -20]]
cons = [-350, -400]

res = linprog(maxim, A_ub=coef, b_ub=cons, bounds=((0,1000),(0,1000),(0,1000)))
print('Minimum cost: ', res.fun)
print('Orders from supplier1: ', res.x[0])
print('Orders from supplier2: ', res.x[1])
print('Orders from supplier3: ', res.x[2])
