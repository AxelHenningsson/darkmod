import sympy

if __name__ == "__main__":
    x_d, y_d, b, nu = sympy.symbols("x_d y_d b nu")

    u_x = -(b / (2 * sympy.pi)) * (
        ((1 - 2 * nu) / (4 * (1 - nu))) * sympy.atan(y_d / x_d)
        + ((x_d * y_d) / (2 * (1 - nu) * (x_d**2 + y_d**2)))
    )

    u_y = -(b / (2 * sympy.pi)) * (
        ((1 - 2 * nu) / (4 * (1 - nu))) * sympy.ln(x_d**2 + y_d**2)
        + ((x_d**2 - y_d**2) / (4 * (1 - nu) * (x_d**2 + y_d**2)))
    )

    print(u_x)
    print(u_y)

    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    b = sympy.Symbol("b")
    nu = sympy.Symbol("nu")

    u_x = (b / (2 * sympy.pi)) * (
        1 / (sympy.tan((y / x))) + (x * y / (2 * (1 - nu) * (x * x + y * y)))
    )
    du_dx = sympy.Derivative(u_x, x).simplify()

    u_x = (b / (2 * sympy.pi)) * (
        1 / (y / x) + (x * y / (2 * (1 - nu) * (x * x + y * y)))
    )
    sympy.Derivative(u_x, x).simplify()
