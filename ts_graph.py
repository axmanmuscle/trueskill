import numpy as np
from scipy.stats import norm

class Gaussian():

    #: Precision, the inverse of the variance.
    pi = 0
    #: Precision adjusted mean, the precision multiplied by the mean.
    tau = 0

    def __init__(self, pi=0, tau=0):
        # pi = sigma ** -2
        # tau = pi * mu
        self.pi = pi
        self.tau = tau

    @property
    def mu(self):
        """A property which returns the mean."""
        return self.pi and self.tau / self.pi

    @property
    def sigma(self):
        """A property which returns the the square root of the variance."""
        return np.sqrt(1 / self.pi) if self.pi else np.inf

    def __truediv__(self, other):
        pi, tau = self.pi - other.pi, self.tau - other.tau
        return Gaussian(pi=pi, tau=tau)
    
    def __mul__(self, other):
        pi, tau = self.pi + other.pi, self.tau + other.tau
        return Gaussian(pi=pi, tau=tau)

class Variable(Gaussian):

    def __init__(self):
        self.messages = {}
        super(Variable, self).__init__()

    def _delta(self, val):
        """
        difference for update
        """
        pi_delta = abs(self.pi - val.pi)
        if pi_delta == np.inf:
            return 0.
        return max(abs(self.tau - val.tau), np.sqrt(pi_delta))
    
    def set(self, value):
        """
        set a new value for the variable and record the delta of the change
        """
        delta = self._delta(value)
        self.pi, self.tau = value.pi, value.tau
        return delta

    def update_val(self, factor, pi=0, tau=0, Value = None):
        """
        TODO: if we directly update our marginal, we need to change our message as well
        """
        if Value:
            value = Value
        else:
            value = Gaussian(pi=pi, tau=tau)

        # update message
        old_message = self.messages[factor]
        self.messages[factor] = value * old_message / self

        # update value
        delta = self.set(value)
        return delta
    
    def update_message(self, factor, pi=0, tau=0, Message = None):
        """
        TODO: when we update the message we have to update our current marginal as well
        """
        if Message:
            message = Message
        else:
            message = Gaussian(pi=pi, tau=tau)
        
        # update message
        old_message = self.messages[factor]
        self.messages[factor] = message

        # update value
        delta = self.set(self / old_message * message)
        return delta

    def get_message(self, factor):
        return self.messages[factor]
    
    def get_diff(self, factor):
        diff = Gaussian(pi = self.pi - self.messages[factor].pi, tau = self.tau - self.messages[factor].tau)
        return diff
    
    def __repr__(self):
        s = "Variable with mean {} and sd {}".format(self.mu, self.sigma)
        return(s)
    
    def __getitem__(self, factor):
        return self.messages[factor]

    def __setitem__(self, factor, message):
        self.messages[factor] = message
    
    
class PriorFactor():

    def __init__(self, variable, value, dynamic):
        self.value = value # this is the value of the node, a gaussian RV
        self.var = variable # this is the variable the node points to

        self.dynamic = dynamic
        variable[self] = Gaussian()

    def down(self):
        """
        something
        """
        sigma = np.sqrt(self.value.sd**2 + self.dynamic**2)
        pi = sigma**-2
        tau = pi * self.value.mean
        value = Gaussian(pi, tau)
        return self.var.update_val(self, value.pi, value.tau)
    
    # def __repr__(self):
    #     return 'prior factor with {} mean'.format(self.value.mu)

class LikelihoodFactor():
    """
    likelihood node looks like

                   -----
                   | y |
                   -----
                     |
                     v
              ---------------
              | N(x; y, c^2) |
              ---------------
                     |
                     v
                   -----
                   | x |
                   -----

    so x is our "value var" and y is our "mean var"
    update is same up and down (since N(x; y, c) = N(y; x, c))
    """
    
    def __init__(self, mean_var, value_var, variance):
        self.mean = mean_var
        self.value = value_var
        self.variance = variance

        # set up messages
        mean_var[self] = Gaussian()
        value_var[self] = Gaussian()

    def calc_a(self, diff):
        return 1. / (1 + self.variance * diff.pi)
    
    def down(self):
        diff_y = self.mean.get_diff(self)
        a = self.calc_a(diff_y)
        pi = a * diff_y.pi
        tau = a * diff_y.tau
        newval = self.value.update_message(self, pi, tau)
        return newval

    def up(self):
        diff_x = self.value.get_diff(self)
        a = self.calc_a(diff_x)
        pi = a * diff_x.pi
        tau = a * diff_x.tau
        newval = self.mean.update_message(self, pi, tau)
        return newval
    
class SumFactor():
    """
    Node represents a combination of the inputs weighted by coefficients into an output:
    x = a^Ty
    """

    def __init__(self, sum_var, inputs, coeffs):
        self.sum = sum_var
        self.inputs = inputs
        self.coeffs = coeffs

        # set up messages
        sum_var[self] = Gaussian()
        for i in inputs:
            i[self] = Gaussian()

    def down(self):
        vals = self.inputs
        #msgs = [var.get_message() for var in vals]
        out = self.update(self.sum, vals, self.coeffs)
        return out
    
    def up(self, index=0):
        coeff = self.coeffs[index]
        coeffs = []
        for x, c in enumerate(self.coeffs):
            try:
                if x == index:
                    coeffs.append(1. / coeff)
                else:
                    coeffs.append(-c / coeff)
            except ZeroDivisionError:
                coeffs.append(0.)
        vals = self.inputs[:]
        vals[index] = self.sum

        out = self.update(self.inputs[index], vals, coeffs)
        return out

    def update(self, sum_var, inputs, coeffs):
        pi_inv = 0
        tau = 0
        for input, coeff in zip(inputs, coeffs):
            diff = input.get_diff(self)
            tau += coeff * diff.mu
            if pi_inv == np.inf:
                continue
            try:
                # numpy.float64 handles floating-point error by different way.
                # For example, it can just warn RuntimeWarning on n/0 problem
                # instead of throwing ZeroDivisionError.  So div.pi, the
                # denominator has to be a built-in float.
                pi_inv += coeff ** 2 / float(diff.pi)
            except ZeroDivisionError:
                pi_inv = np.inf
        pi = 1./pi_inv
        tau *= pi
        out = sum_var.update_message(self, pi, tau)

        return out

class TruncateFactor():

    def __init__(self, var, v_func, w_func, draw_margin):
        self.var = var
        self.v_func = v_func
        self.w_func = w_func
        self.draw_margin = draw_margin

        # add message
        var[self] = Gaussian()

    def up(self):
        var = self.var
        div = var.get_diff(self)
        sqrt_pi = np.sqrt(div.pi)
        v = self.v_func(div.tau / sqrt_pi, self.draw_margin * sqrt_pi)
        w = self.w_func(div.tau / sqrt_pi, self.draw_margin * sqrt_pi)

        denom = 1.-w
        pi = div.pi / denom
        tau = (div.tau + sqrt_pi * v) / denom
        out = var.update_val(self, pi=pi, tau=tau)

        return out


def v_win(x, draw_margin):
    """
    win version of v_function, we probably won't need the draw version
    """
    diff = x - draw_margin
    denom = norm.cdf(diff)
    return (norm.pdf(diff) / denom) if denom else -diff

def w_win(x, draw_margin):
    """
    win version of w_function, we probably won't need the draw version
    """
    v = v_win(x, draw_margin)
    return v*(v + x - draw_margin)