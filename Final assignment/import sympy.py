import sympy

s = sympy.Symbol('s', complex=True)

A = sympy.Matrix([
    [-10.5,   0.5,    0,      0,     0,    0,    0,    5],
    [ 10.5,  -11,     0.5,    0,     0,    0,    0,    0],
    [  0,    10.5,  -11,     0.5,    0,    0,    0,    0],
    [  0,     0,    10.5,  -10.5,   0,    0,    0,    0],
    [  0,     0,     0,     20,    -20,   0,    0,    0],
    [  0,     0,     0,      0,    20,   -20,   0,    0],
    [  0,     0,     0,      0,     0,    20,  -20,   0],
    [  0,     0,     0,      0,     0,     0,   20,  -20],
])
B = sympy.Matrix([5, 0, 0, 0, 0, 0, 0, 0])
C = sympy.Matrix([[0, 0, 0, 1, 0, 0, 0, 0]])  # picks out x4
D = sympy.Integer(0)

# Transfer function G(s) = C (sI - A)^(-1) B + D
I = sympy.eye(A.shape[0])
G_s_sym = C*( (s*I - A).inv() )*B + D
G_s_sym = sympy.simplify(G_s_sym[0])  # It's a 1x1, so pull out scalar

print("Exact symbolic transfer function, G(s) =")
sympy.pretty_print(G_s_sym)

# Evaluate DC gain at s=0
DC_gain = G_s_sym.subs(s, 0)
print(f"\nDC gain (G(0)) = {DC_gain}")

# Let's do expansions of various orders
for order in [2,4,6,8,10,12,14,20]:
    series_approx = sympy.series(G_s_sym, s, 0, order)
    print(f"\nSeries expansion up to s^{order-1}:")
    sympy.pretty_print(series_approx)
