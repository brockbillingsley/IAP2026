"""
Equation numbers refer to
https://arxiv.org/pdf/2503.22549
"""

import zfit
from zfit import z

# Import angular functions to be used in the classes
import angularfunctions as af


class my2Dpdf(zfit.pdf.ZPDF):
    # 2D angular pdf to fit Eq. (1)
    _PARAMS = ['App', 'A0', 'AS', 'Aqs', 'Aqc',
               'AfbHS', 'AfbHC', 'AfbLS', 'AfbLC']

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)  # Unstack the two angular variables

        # Retrieve the parameters
        App = self.params['App']
        A0 = self.params['A0']
        AS = self.params['AS']
        Aqs = self.params['Aqs']
        Aqc = self.params['Aqc']
        AfbHS = self.params['AfbHS']
        AfbHC = self.params['AfbHC']
        AfbLS = self.params['AfbLS']
        AfbLC = self.params['AfbLC']

        return af.fun_2D(cosh, cosl, AS, App, A0, Aqs, Aqc, AfbHS, AfbHC, AfbLS, AfbLC)


def integral(limits, params, model):
    # Integral of the 2D angular pdf over the given limits
    del model

    lower, upper = limits.v1.limits

    App = params['App']
    A0 = params['A0']
    AS = params['AS']
    Aqc = params['Aqc']
    Aqs = params['Aqs']
    AfbHS = params['AfbHS']
    AfbHC = params['AfbHC']
    AfbLS = params['AfbLS']
    AfbLC = params['AfbLC']

    Int = 0

    for f, a in zip([af.int_App, af.int_A0, af.int_AS, af.int_Aqc, af.int_Aqs, af.int_AfbHS, af.int_AfbHC, af.int_AfbLS, af.int_AfbLC], [App, A0, AS, Aqc, Aqs, AfbHS, AfbHC, AfbLS, AfbLC]):
        Int += (f(upper[0], upper[1]) - f(upper[0],lower[1]) - f(lower[0], upper[1]) + f(lower[0],lower[1])) * a
    return Int


class my2Dpdf_AS(zfit.pdf.ZPDF):
    # 2D angular pdf for the S-wave contribution
    # Third term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return (3/8) * (1-cosl*cosl)


def integral_AS(limits, params, model):
    # Integral of the S-wave contribution over the given limits
    del model

    lower, upper = limits.v1.limits

    return af.int_AS(upper[0], upper[1]) - af.int_AS(upper[0], lower[1]) - af.int_AS(lower[0], upper[1]) + af.int_AS(lower[0], lower[1])


class my2Dpdf_App(zfit.pdf.ZPDF):
    # 2D angular pdf for the perp/parallel contribution
    # First term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_App(cosh, cosl)


def integral_App(limits, params, model):
    # Integral of the perp/parallel contribution over the given limits
    del model

    lower, upper = limits.v1.limits

    return af.int_App(upper[0], upper[1]) - af.int_App(upper[0], lower[1]) - af.int_App(lower[0], upper[1]) + af.int_App(lower[0], lower[1])


class my2Dpdf_Aq(zfit.pdf.ZPDF):
    # 2D angular pdf for the beta-dependent contribution
    # Fourth term in Eq. (1)
    # Split into two parts for numerical stability
    _PARAMS = ['Aqc', 'Aqs']

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        rAqs = 1/(1 + self.params['Aqc']/self.params['Aqs'])
        rAqc = 1/(1 + self.params['Aqs']/self.params['Aqc'])

        return af.fun_Aqc(cosh, cosl) * rAqc + af.fun_Aqs(cosh, cosl) * rAqs


def integral_Aq(limits, params, model):
    # Integral of the beta-dependent contribution over the given limits
    del model

    lower, upper = limits.v1.limits

    rAqs = 1/(1 + params['Aqc']/params['Aqs'])
    rAqc = 1/(1 + params['Aqs']/params['Aqc'])

    IntAqs = af.int_Aqs(upper[0], upper[1]) - af.int_Aqs(upper[0],lower[1]) - af.int_Aqs(lower[0], upper[1]) + af.int_Aqs(lower[0],lower[1])
    IntAqc = af.int_Aqc(upper[0], upper[1]) - af.int_Aqc(upper[0],lower[1]) - af.int_Aqc(lower[0], upper[1]) + af.int_Aqc(lower[0],lower[1])

    return IntAqs * rAqs + IntAqc * rAqc


class my2Dpdf_A0(zfit.pdf.ZPDF):
    # 2D angular pdf for the A0 contribution
    # Second term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_A0(cosh, cosl)


def integral_A0(limits, params, model):
    # Integral of the A0 contribution over the given limits
    del model

    lower, upper = limits.v1.limits

    return af.int_A0(upper[0], upper[1]) - af.int_A0(upper[0],lower[1]) - af.int_A0(lower[0], upper[1]) + af.int_A0(lower[0],lower[1])


class my2Dpdf_AfbHS(zfit.pdf.ZPDF):
    # 2D angular pdf for the asymemtry term on the hadronic side (costhetah) with sin2thetal
    # Sixth term in Eq. (1)

    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_AfbHS(cosh, cosl)


class my2Dpdf_AfbHC(zfit.pdf.ZPDF):
    # 2D angular pdf for the asymemtry term on the hadronic side (costhetah) with cos2thetal
    # Fifth term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_AfbHC(cosh, cosl)


class my2Dpdf_AfbLS(zfit.pdf.ZPDF):
    # 2D angular pdf for the asymemtry term on the leptonic side (costhetal) with sin2thetah
    # Eighth term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_AfbLS(cosh, cosl)


class my2Dpdf_AfbLC(zfit.pdf.ZPDF):
    # 2D angular pdf for the asymemtry term on the leptonic side (costhetal) with cos2thetah
    # Seventh term in Eq. (1)
    _PARAMS = []

    def _unnormalized_pdf(self, x):
        cosh, cosl = z.unstack_x(x)

        return af.fun_AfbLC(cosh, cosl)
