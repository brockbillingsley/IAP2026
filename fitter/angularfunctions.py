import numpy as np
import zfit.z.numpy as znp

# For all functions:
# x: cos(theta_h)
# y: cos(theta_l)


def fun_App(x, y):
    # Perp/parallel function
    return (9/32) * (1 - x**2) * (1+y**2)


def int_App(x, y):
    # Perp/parallel integral
    return (9/32) * (x - x**3/3) * (y+y**3/3)


def fun_App_cosh(x):
    # Perp/parallel function (cosh projection)
    return (3/4) * (1 - x**2)


def fun_A0(x, y):
    # 0 function
    return (9/8) * x**2 * (1-y**2)


def int_A0(x, y):
    # 0 integral
    return (9/8) * (x**3/3) * (y-y**3/3)


def fun_A0_cosh(x):
    # 0 function (cosh projection)
    return (3/2) * x**2


def fun_Aqs(x, y):
    # beta-dependent function part 1
    return (3/8) * (1-x**2)


def int_Aqs(x, y):
    # beta-dependent integral part 1
    return (3/8) * (x-x**3/3) * y


def fun_Aqc(x, y):
    # beta-dependent function part 2
    return (3/4) * x**2


def int_Aqc(x, y):
    # beta-dependent integral part 2
    return (3/4) * x**3/3 * y


def fun_AS(x, y):
    # S-wave function
    return (3/8) * (1-y**2)


def int_AS(x, y):
    # S-wave integral
    return (3/8) * x * (y-y**3/3)


def fun_AS_cosh(x):
    # S-wave function (cosh projection)
    return (1/2) * znp.ones_like(x)


def fun_AfbHS(x, y):
    # Asymmetry term on the hadronic side (costhetah) with sin2thetal
    return x * (1-y**2)


def int_AfbHS(x, y):
    # Asymmetry term on the hadronic side (costhetah) with sin2thetal integral
    return x**2/2 * (y-y**3/3)


def proj_AfbHS(x, n):
    # Projection of the asymmetry term on the hadronic side (costhetah) with sin2thetal
    if n == "cosh":
        return (4/3) * x
    if n == "cosl":
        return np.zeros(len(x))


def fun_AfbHC(x, y):
    # Asymmetry term on the hadronic side (costhetah) with cos2thetal
    return x * y**2


def int_AfbHC(x, y):
    # Asymmetry term on the hadronic side (costhetah) with cos2thetal integral
    return x**2/2 * (y**3/3)


def proj_AfbHC(x, n):
    # Projection of the asymmetry term on the hadronic side (costhetah) with cos2thetal
    if n == "cosh":
        return (2/3) * x
    if n == "cosl":
        return np.zeros(len(x))


def fun_AfbLS(x, y):
    # Asymmetry term on the leptonic side (costhetal) with sin2thetah
    return (1-x*x) * y


def int_AfbLS(x, y):
    # Asymmetry term on the leptonic side (costhetal) with sin2thetah integral
    return (x-x**3/3) * (y**2/2)


def proj_AfbLS(x, n):
    # Projection of the asymmetry term on the leptonic side (costhetal) with sin2thetah
    if n == "cosh":
        return np.zeros(len(x))
    if n == "cosl":
        return (4/3) * x


def fun_AfbLC(x, y):
    # Asymmetry term on the leptonic side (costhetal) with cos2thetah
    return x*x * y


def int_AfbLC(x, y):
    # Asymmetry term on the leptonic side (costhetal) with cos2thetah integral
    return x**3/3 * (y**2/2)


def proj_AfbLC(x, n):
    # Projection of the asymmetry term on the leptonic side (costhetal) with cos2thetah
    if n == "cosh":
        return np.zeros(len(x))
    if n == "cosl":
        return (2/3) * x


def fun_2D(cosh, cosl, AS, App, A0, Aqs, Aqc, AfbHS, AfbHC, AfbLS, AfbLC):
    # Full 2D angular function
    return fun_AS(cosh, cosl) * AS + fun_App(cosh, cosl) * App + fun_A0(cosh, cosl) * A0 \
            + fun_Aqs(cosh, cosl) * Aqs + fun_Aqc(cosh, cosl) * Aqc \
            + fun_AfbLS(cosh, cosl) * AfbLS + fun_AfbLC(cosh, cosl) * AfbLC \
            + fun_AfbHS(cosh, cosl) * AfbHS + fun_AfbHC(cosh, cosl) * AfbHC
