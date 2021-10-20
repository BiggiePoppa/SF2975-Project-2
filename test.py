import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import scipy
from matplotlib import gridspec

class HullWhite():
    def __init__(self, times, yield_curve, alpha, sigma, r0 = None):

        self.dt = 0.00001
        self.times = times
        self.alpha = alpha
        self.sigma = sigma
        if not r0:
            self.r0 = yield_curve.iloc[0]
        else:
            self.r0 = r0
        self.yieldCurve = splrep(times, yield_curve) #create a spline representation of the yield curve
    
        self.spline_theta()


    def y(self, t):
        '''
        Returns y(t) using the spline reperesntation
        '''
        return splev(t, self.yieldCurve)

    def dydt(self, t):
        '''
        returns dy/dt using the spline representation
        '''
        return splev(t, self.yieldCurve, der = 1)


    def forward_rate(self, t):
        '''
        returns the forward rate, i.e f(0,T), as a function of the yield-curve 
        '''
        return self.y(t) + t * self.dydt(t)

    def theta(self, t):
        '''
        calculates theta for a given T
        '''
        dfdt = self.forward_rate(max(t + self.dt, 0)) - self.forward_rate(max(t-self.dt, 0))
        dfdt /= (2*self.dt)

        return dfdt + self.alpha * self.forward_rate(t) + (self.sigma**2)/(self.alpha*2)*(1-np.exp(-2*self.alpha*t))

    def spline_theta(self):
        '''
        Create a spline representation of theta
        '''
        times = np.linspace(self.times.iloc[0], self.times.iloc[-1], 1000)
        thetas = [self.theta(t) for t in times]
        self.theta_splrep = splrep(times, thetas)

    def eval_theta(self, x):
        '''
        method for evaluating the spline reprsentation of theta(t), returns theta(t)
        '''
        return splev(x, self.theta_splrep)

    def A(self, t, T):
        '''
        A(t,T)
        '''
        def int1(x):
            theta = splev(x, self.theta_splrep)
            return theta * self.B(x,T)
        def int2(x):
            return self.B(x, T)**2
        
        eval1 = scipy.integrate.quad(int1, t, T)[0]
        eval2 = scipy.integrate.quad(int2, t, T)[0]

        return -eval1 + 0.5*(self.sigma**2)*eval2

    def B(self, t, T):
        '''
        B(t,T)
        '''
        return (1/self.alpha) * (1-np.exp(-self.alpha*(T-t)))
    
    def simple_zcb(self, T):
        B = self.B(0, T)
        A = self.A(0, T)
        return np.exp(A-self.r0*B)

    def simulate_paths(self, number_of_steps, number_of_simulations, T):
        dt = T/float(number_of_steps)

        xh = np.zeros((number_of_steps+ 1, number_of_simulations))
        self.rates = np.zeros_like(xh)
        self.rates_times = np.linspace(0, T, number_of_steps + 1)

        xh[0] = self.r0

        for i in range(1, number_of_steps + 1):
            xh[i] = xh[i-1] + (self.theta(self.rates_times[i-1]) - self.alpha * xh[i-1]) * dt \
                             + self.sigma * np.sqrt(dt) * np.random.standard_normal(number_of_simulations)
        self.rates = xh


    def price_zcb_using_rates(self, t, T):
        '''
        Uses simulated rates to calculate p(t,T)
        '''
        times  = self.rates_times
        dt = times[1] - times[0] 

        idx = np.where((times < T) & (times > t ))[0]
        rates = self.rates[idx,:]

        mean_p_t_T = np.exp(- rates.sum(axis = 0) * dt).mean()
        std_p_t_T = np.exp(- rates.sum(axis = 0) * dt).std()

        return mean_p_t_T, std_p_t_T


