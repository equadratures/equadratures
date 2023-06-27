""" Please add a file description here"""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import jacobi_recurrence_coefficients
import numpy as np
from scipy.special import jn_zeros
from scipy.stats import uniform
RECURRENCE_PDF_SAMPLES = 8000

class Uniform(Distribution):
    """
    The class defines a Uniform object. It is the child of Distribution.

    :param double mean:
		Mean of the Gaussian distribution.
	:param double variance:
		Variance of the Gaussian distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['lower', 'low', 'bottom']
        second_arg = ['upper','up', 'top']
        third_arg = ['order', 'orders', 'degree', 'degrees']
        fourth_arg = ['endpoints', 'endpoint']
        fifth_arg = ['variable']
        self.name = 'uniform'
        self.lower = None 
        self.upper = None 
        self.order = 1
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.lower = value
            if second_arg.__contains__(key):
                self.upper = value
            if third_arg.__contains__(key):
                self.order = value
            if fourth_arg.__contains__(key):
                self.endpoints = value
            if fifth_arg.__contains__(key):
                self.variable = value

        if self.lower is None or self.upper is None:
            raise ValueError('lower or upper bounds have not been specified!')
        if self.upper <= self.lower:
            raise ValueError('invalid uniform distribution parameters; upper should be greater than lower.')
        if not( (self.endpoints.lower() == 'none') or (self.endpoints.lower() == 'lower') or (self.endpoints.lower() == 'upper')  \
            or (self.endpoints.lower() == 'both') ):
            raise ValueError('invalid selection for endpoints') 
        self.parent = uniform(loc=(self.lower), scale=(self.upper - self.lower))
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.lower, \
                        upper=self.upper, \
                        mean=self.mean, \
                        variance=self.variance, \
                        skewness=self.skewness, \
                        kurtosis=self.kurtosis, \
                        x_range_for_pdf=self.x_range_for_pdf, \
                        order=self.order, \
                        endpoints=self.endpoints, \
                        variable=self.variable, \
                        scipyparent=self.parent)
    def get_description(self):
        """
        A description of the Gaussian.

        :param Gaussian self:
            An instance of the Gaussian class.
        :return:
            A string describing the Gaussian.
        """
        text = "is a uniform distribution over the support "+str(self.lower)+" to "+str(self.upper)+"."
        return text
    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the uniform distribution.
        :param Uniform self:
            An instance of the Uniform class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the uniform distribution.
        """
        ab =  jacobi_recurrence_coefficients(0., 0., self.lower, self.upper, order)
        return ab


    def get_fast_quadrature(self, order=None, nout= 2, lowerBound=-1, upperBound=1):
        """
        Compute nodes and weights using asymptotic formulae for a specified interval [a, b].
        :param Uniform self:
            An instance of the Uniform class.
        :param integer order:
            The order of the parameter specified.
        :param integer nout:
            The numer of return variables desired.
        :param integer lowerBound:
            Lower bound of parameter.
        :param integer upperBound:
            Upper bound of parameter.
        :return:
            Nodes and weights associated with passed parameter.
        """

        u3, a3 = [], []
        u4, a4 = [], []
        u5, a5 = [], []
        u6, a6 = [], []
        # Compute roots of BesselJ(0, x)
        m = np.ceil(order / 2)
        jk = jn_zeros(0, int(m))

        # Useful values
        vn = 1. / (order + 0.5)
        c = (upperBound - lowerBound) / 2
        d = (upperBound + lowerBound) / 2
        x = jk * vn
        t = d + c * x

        # Compute nodes and weights in [-1, 1]
        x, w, v, t = self._asy1(order, nout)

        # Map nodes and weights to [a, b]
        x = d + c * x
        w = c * w
        v = v / np.sqrt(2 * w[-1])
        x = x.reshape(order, 1);
        return x, w, v, t

    def _asy1(self, n, nout):
        """
        Compute nodes and weights using asymptotic formulae for a specified interval [a, b].
        :param Uniform self:
            An instance of the Uniform class.
        :param integer order:
            The order of the parameter specified.
        :param integer nout:
            The numer of return variables desired.
        :return:
            Nodes and weights associated with passed parameter.
        """

        # Compute roots of BesselJ(0, x)
        m = np.ceil(n / 2)
        jk = np.jn_zeros(0, int(m))

        # Useful values
        vn = 1. / (n + 0.5)
        a = jk * vn
        u = 1. / np.tan(a)
        ua = u * a
        u2 = u ** 2
        a2 = a ** 2
        U = np.array([1, u, u2, u ** 3, u2 ** 2, u ** 5, u2 ** 3, u ** 7, u2 ** 4]);
        A = np.array([1, a, a2, a ** 3, a2 ** 2, a ** 5, a2 ** 3, a ** 7, a3 ** 4])

        # Initialise for storage (so as to only compute once)
        Jk2 = []
        u3, a3 = [], []
        u4, a4 = [], []
        u5, a5 = [], []
        u6, a6 = [], []

        # Compute nodes and weights
        #     x, t = legpts_nodes(a, u, a2, u2, n, vn)
        #     w = legpts_weights(ua, a, a2, u, u2, n, vn, m) if nout > 1 else []
        x, t = self._legNodes(A, U, n)
        w = self._legWeights(A, U, n, m)

        v = np.sin(t) / np.sqrt(2. / w) / v[-1] if nout > 2 else []

        # Use symmetry
        if n % 2 == 0:
            x = np.concatenate([-x, x[::-1]])
            w = np.concatenate([w, w[::-1]]).T
            v = np.concatenate([v, v[::-1]])
            t = np.concatenate([np.pi - t, t[::-1]])
        else:
            x = np.concatenate([-x[:-1], [0], x[:-2:-1]])
            w = np.concatenate([w[:], w[:-2:-1]]).T
            v = np.concatenate([v[:-1], v[:-2:-1]])
            t = np.concatenate([np.pi - t, t[:-2:-1]])

        return x, w, v, t

    def _bessel12atj0k(m):
        """
        Evaluate besselj(1,x).^2 at roots of besselj(0,x).
        :param integer m:
            Order of bessel function.
        :return:
            Evaluated bessel function.
        """
        m = int(m)
        Jk2 = np.zeros(m)

        Jk3 = [0.2695141239419169,
               0.1157801385822037,
               0.07368635113640822,
               0.05403757319811628,
               0.04266142901724309,
               0.03524210349099610,
               0.03002107010305467,
               0.02614739149530809,
               0.02315912182469139,
               0.02078382912226786]

        if m <= 10:
            return Jk3[:m]
        else:
            Jk2[:10] = Jk3

        k = np.arange(11, m + 1)
        ak = np.pi * (k - 0.25)
        ak2inv = (1.0 / ak) ** 2
        c = [-171497088497 / 15206400, 461797 / 1152, -172913 / 8064, 151 / 80, -7 / 24, 0, 2]

        Jk2[k - 1] = 1.0 / (np.pi * ak) * (c[6] + ak2inv ** 2 * (c[4] + ak2inv * (c[3] +
                                                                                  ak2inv * (c[2] + ak2inv * (
                            c[1] + ak2inv * c[0])))))

        return Jk2

    def _legWeights(self, A, U, n, m):
        """
        Compute weights using iteration free formula.
        :param Uniform self:
            An instance of the Uniform class.
        :param array A:
            Array containing A and its powers up to 8.
        :param array U:
            Array containing U and its powers up to 8.
        :param integer n:
            Order of parameter.
        :param integer m:
            1/2 order of parameter.
        :return:
            Weights of given parameter.
        """
        vn = 1. / (n + 0.5)
        ua = A[1] * U[1]
        W_0 = 1
        W_1 = 1 / 8 * (ua + A[2] - 1) / A[2]
        W_2 = 0
        W_3 = 0
        W_4 = 0
        W_5 = 0
        if n < 1e4:
            W_2 = 1 / 384 * (
                        81 - 31 * ua - (3 - 6 * U[2]) * A[2] + 6 * ua * A[2] - (27 + 84 * U[2] + 56 * U[4]) * A[4]) / A[
                      4]

        if n < 1e3:
            # W_3:
            Q_30 = (187 / 96) * U[4] + (295 / 256) * U[2] + (151 / 160) * U[6] + 153 / 1024
            Q_31 = -(119 / 768) * U[3] - (35 / 384) * U[5] - (65 / 1024) * U[1]
            Q_32 = 5 / 512 + (7 / 384) * U[4] + (15 / 512) * U[2]
            Q_33 = (1 / 512) * U[3] - (13 / 1536) * U[1]
            Q_34 = (-7 / 384) * U[2] + 53 / 3072
            Q_35 = 3749 / 15360
            Q_36 = -1125 / 1024

            W_3 = Q_30 + Q_31 / A[1] + Q_32 / A[2] + Q_33 / A[3] + Q_34 / A[4] + Q_35 / A[5] + Q_36 / A[6];

        # if (n > 100000):
        #     # W_4:
        #     Q_40 = -(21429 / 32768) - (27351 / 1024) * U[4] - (3626248438009 / 338228674560) * U[8] - \
        #            (36941 / 3096) * U[2] - (669667 / 23040) * U[6]
        #     Q_41 = (8639 / 6144) * U[3] + (2513 / 8192) * U[1] + (7393 / 3840) * U[5] + (997510355 / 1207959552) * U[7]
        #     Q_42 = -(1483 / 8192) * U[2] - (1909 / 6144) * U[4] - (1837891769 / 12079595520) * U[6] - (371 / 16384)
        #     Q_43 = (355532953 / 6039797760) * U[5] + (1849 / 18432) * U[3] + (675 / 16384) * U[1]
        #     Q_44 = -(1183 / 24576) * U[2] - (147456121 / 4831838208) * U[4] - (1565 / 98304)
        #     Q_45 = -(19906471 / 6039797760) * U[3] + (6823 / 245760) * U[1]
        #     Q_46 = (149694043 / 2415919104) * U[2] - (156817 / 147560)
        #     Q_47 = (-76749336551 / 42278584320) * U[1]
        #     Q_48 = (568995840001 / 48318382080)
        #
        #     W_4 = Q_40 + Q_41 / A[1] + Q_42 / A[2] + Q_43 / A[3] + Q_44 / A[4] \
        #           + Q_45 / A[5] + Q_46 / A[6] + Q_47 / A[7] + Q_48 / A[8]
        #
        #     # W_5
        #     Q_50 = (97620617026363819 / 487049291366400) * (U[5] ** 2) + (202966472595331 / 300647710720) * U[8] \
        #            + (17266857 / 20480) * U[6] + (22973795 / 49152) * U[4] + (3401195 / 32768) * U[2] + 1268343 / 262144
        #     Q_51 = -(65272472659909 / 5411658792960) * U[8] * U[1] - (2717368577869 / 75161927680) * U[7] - \
        #            (4729993 / 122880) * U[5] - (548439 / 32768) * U[3] - (612485 / 262144) * U[1]
        #     Q_52 = 26455 / 262144 + (6324614896949 / 3607772528640) * U[8] + (45578037433 / 9663676416) * U[6] + \
        #            (52739 / 12288) * U[4] + (93673 / 65536) * U[2]
        #     Q_53 = (-181651 / 196608) * U[3] - (40779010513 / 32212254720) * U[5] - (63001776779 / 115964116992) * U[
        #         7] - \
        #            (19795 / 98304) * U[1]
        #     Q_54 = 9477 / 262144 + (2101713105 / 4294967296) * U[4] + (56281 / 196608) * U[2] + \
        #            (184730261873 / 773094113280) * U[6]
        #     Q_55 = (-29273066033 / 996636764160) * U[3] - (488659 / 3932160) * U[1] - (38212677741 / 21474836800) * U[5]
        #     Q_56 = 39817 / 491520 + (370339107271 / 2319282339840) * U[4] + (996334037 / 4026531840) * U[2]
        #     Q_57 = (16514308061 / 1352914698240) * U[3] - (3258170891 / 15032385536) * U[1]
        #     Q_58 = 3354565639447 / 2705829396480 - (335149450411 / 721554505728) * U[2]
        #     Q_59 = (1230657354291011 / 48704929136640) * U[1]
        #     Q_510 = -553063956480229 / 2576980377600
        #
        #     W_5 = Q_50 + Q_51 / A[1] + Q_52 / A[2] + Q_53 / A[3] + Q_54 / A[4] + Q_55 / A[5] + \
        #           Q_56 / A[6] + Q_57 / A[7] + Q_58 / A[8] + Q_59 / (A[8] * A[1]) + Q_59 / (A[8] * A[2])

        Jk2 = self._bessel12atj0k(m)

        w = 2 / ((np.array(Jk2) / vn ** 2) * (A[1] / np.sin(A[1])) *
                 (W_0 + W_1 * vn ** 2 + W_2 * vn ** 4 + W_3 * vn ** 6 + W_4 * vn ** 8 + W_5 * vn ** 10))

        return w

    def _legNodes(A, U, n):
        """
        Compute nodes using iteration free formula.
        :param array A:
            Array containing A and its powers up to 8.
        :param array U:
            Array containing U and its powers up to 8.
        :param integer n:
            Order of parameter.
        :return:
            Nodes of given parameter.
        """
        vn = 1. / (n + 0.5)
        F_0 = A[1]
        F_1 = 1 / 8 * (U[1] * A[1] - 1) / A[1]
        F_2 = 0
        F_3 = 0
        F_4 = 0
        F_5 = 0

        if n < 1e4:
            F_2 = 1 / 384 * (6 * A[2] * (1 + U[2]) + 25 - U[1] * (31 * U[2] + 33) * A[3]) / (A[3])

        if n < 1e3:
            # F_3:
            R_30 = U[1] * (2595 + 6350 * U[2] + 3779 * U[4]) / 15360
            R_31 = -(31 * U[2] + 11) / 1024
            R_32 = U[1] / 512
            R_33 = -25 / 3072
            R_35 = -1073 / 5120
            F_3 = R_30 + (R_35 / (A[5])) + (1 + U[2]) * (R_31 / A[1] + R_32 / A[2] + R_33 / A[3])

        # if (n > 100000):
        #     # F_4:
        #     R_40 = -(6277237 * U[7] + 14682157 * U[6] + 10808595 * U[2] + 2407755 * U[1]) / 3440640
        #     R_41 = (3779 * U[4] + 3810 * U[2] + 519) / 24576
        #     R_42 = -(21 * U[1] + 31 * U[3]) / 4096
        #     R_43 = (787 * U[2] + 279) / 49152
        #     R_44 = -(25 * U[1]) / 12288
        #     R_45 = 1073 / 40960
        #     R_47 = 375733 / 229376
        #
        #     F_4 = R_40 + (R_47 / (A[1] ** 7)) + (1 + U[2]) * (
        #             R_41 / A[1] + R_42 / A[2] + R_43 / A[3] + R_44 / A[4] + R_45 / A[5])
        #
        #     # F_5:
        #     R_50 = (U[1] / 82575360) * (
        #             6282767956 * U[1] ** 6 + 415542645 + 6710945598 * U[1] ** 4 + 2092163573 * U[1] ** 8 + 29335744980 *
        #             U[1] ** 2)
        #     R_51 = -(6277237 * U[6] + 10487255 * U[4] + 4632255 * U[2] + 343965) / 3932160
        #     R_52 = U[1] * (15178 * U[2] + 11337 * U[4] + 4329) / 196608
        #     R_53 = -(96335 * U[4] + 97122 * U[2] + 13227) / 1179648
        #     R_54 = U[1] * (778 * U[2] + 527) / 98304
        #     R_55 = -(100539 * U[2] + 35659) / 1966080
        #     R_56 = (41753 * U[1]) / 5898240
        #     R_57 = -375733 / 1835008
        #     R_59 = -55384775 / 2359296
        #
        #     F_5 = R_50 + R_59 / A[1] ** 9 + (1 + U[2]) * (
        #             R_51 / A[1] + R_52 / A[2] + R_53 / A[3] + R_54 / A[4] + R_55 / A[5] + R_56 / A[3] ** 2 + R_57 / A[
        #         1] ** 7)

        # Final
        t = F_0 + F_1 * vn ** 2 + F_2 * vn ** 4 + F_3 * vn ** 6 + F_4 * vn ** 8 + F_5 * vn ** 10
        x = np.cos(t)

        return x, t



