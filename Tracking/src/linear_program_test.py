from scipy.optimize import linprog

# obj = [-1, -2]
#
# lhs_ineq = [[ 2,  1],  # Red constraint left side
#             [-4,  5],  # Blue constraint left side
#             [ 1, -2]]
#
# rhs_ineq = [20,  # Red constraint right side
#             10,  # Blue constraint right side
#             2]
# lhs_eq = [[-1, 5]]
#
# rhs_eq = [15]
#
# bnd = [(0, float("inf")),  # Bounds of x
#        (0, float("inf"))] # Bounds of y
#
#
# opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
#             A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
#             method="highs")
# print(opt)


import pulp

# Khởi tạo mô hình
prob = pulp.LpProblem('Bai_toan_san_xuat')

# Định nghĩa biến
x = pulp.LpVariable("x", 0, 1, cat='Integer')
y = pulp.LpVariable("y", 0, 1, cat='Integer')

# Hàm mục tiêu
prob.setObjective(20 * x + 30 * y)

# Ràng buộc
prob.addConstraint(2 * x + y <= 100)
prob.addConstraint(x + 2 * y <= 80)

# Giải bài toán
prob.solve()

# Hiển thị kết quả
print('So luong san pham A:', pulp.value(x))
print('So luong san pham B:', pulp.value(y))
print('Loi nhuan toi da:', pulp.value(prob.objective))