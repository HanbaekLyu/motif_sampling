from sympy import symbols, Matrix, Transpose, sqrt, simplify, series
import progressbar

def path_transform(k):
    # computes path transform of A via path of k nodes
    # k = 0 means k = infty
    x, y = symbols('x y')
    z = (1-x)/2
    w = y*sqrt(x*(1-x)/2)
    a = Matrix([[sqrt(z), sqrt(x), sqrt(z)]])

    D = Matrix([[sqrt(z), 0, 0],
            [0, sqrt(x), 0],
            [0, 0, sqrt(z)]])

    A = Matrix([[z, w, 0],
            [w, x, w],
            [0, w, z]])

    Eval = list(A.eigenvals().keys())
    Evec = A.eigenvects()

    V = Matrix([[-1, 1, 1], [0, 0, 0], [1, 1, 1]])
    for i in range(3):
        v = Evec[i]
        v = v[2]
        v = v[0]
        v = simplify(v)
        V[:, i] = v

    if k == 0:
        v = V[:, 2]
        r = (a*V[:, 2])[0]
        P = (r ** (-2)) * D * v * Transpose(v) * D
    else:
        U = Matrix([[Eval[0] ** (k - 1), 0, 0],
                    [0, Eval[1] ** (k - 1), 0],
                    [0, 0, Eval[2] ** (k - 1)]])
        w = a * V
        t = (w * U * Transpose(w))[0]
        P = (D * V * U * Transpose(V) * D) / t

    P0 = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    bar = progressbar.ProgressBar()
    for i in bar(range(3)):
        for j in range(3):
            q = P[i, j]
            # q = series(q, y, 0, 2)
            P0[i, j] = series(q, x, 0, 2)

    return P0

#  P = path_tranform(0)