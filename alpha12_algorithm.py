import numpy as np
import pylab as pl
import scipy.special as special
from scipy.integrate import quad
from scipy.integrate import fixed_quad

# Set plot parameters to make beautiful plots
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 15
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

#Constants; copy-pasted from symphony
m = 9.1093826e-28
h = 6.6260693e-27
c = 2.99792458e10
e = 4.80320680e-10

n_e     = 1.
theta_e = 3.
theta   = np.pi/4.
B       = 30.

nu_c    = (e * B) / (2. * np.pi * m * c)
omega_c = 2. * np.pi * nu_c

nu    = 5.96362331659 * nu_c
omega = 2. * np.pi * nu

nu_c_dexter = (3./2.) * theta_e**2. * nu_c * np.sin(theta)

#This is the term Df, equation 29 of alpha_12_calculation_outline
def Df(gamma):
    term1 = omega / (m * c**2. * theta_e)
    term2 = 1./ (4. * np.pi * m**3. * c**3. * theta_e * special.kn(2, 1./theta_e))
    ans   = term1 * term2 * np.exp(- gamma / theta_e)
    return ans

#bessel function term integrand, without 1/sin(pi * a) term
def integrand_nopoles(a, p_bar, nu):
    
    #minimum value of a to keep p_perp real; equation 27
    a_min = (nu / nu_c) * (1. - np.cos(theta) * p_bar) / np.sqrt(1. - p_bar**2.)
    
    #check if a > a_min
    if(a <= a_min):
        return 0.    
    
    #gamma in terms of a, p_bar; equation 23
    gamma = (nu_c / nu) * a / (1. - np.cos(theta) * p_bar)
    
    #p_parallel, equation 21
    p_parallel = gamma * m * c * p_bar

    #p_perp, equation 22
    p_perp = m * c * np.sqrt(gamma**2. * (1. - p_bar**2.) - 1.)
    
    #this is gamma as a function of p_perp and p_parallel, equation 28
    gamma_check = np.sqrt((p_perp/(m * c))**2. + (p_parallel/(m * c))**2. + 1.)
    
    #Omega, equation 8
    Omega = omega_c / gamma
    
    #z, equation 5
    z = omega * np.sin(theta) * p_perp / (gamma * m * c * Omega) 
    
    #M, equation 6
    M = (np.cos(theta) - (p_parallel / (gamma * m * c))) / np.sin(theta)

    #N, equation 7
    N = p_perp / (gamma * m * c)

    #bessel function term in integrand (eq. 16); note that the 1/sin(pi * a) is taken out
    # so that we can use the Weiss method
    bessel_term = np.pi * special.jv(a, z) * special.jvp(-a, z)
    
    if(np.isnan(bessel_term) == True):
        bessel_term = - 1./z * np.sin(np.pi * a)
#        return 0.
      
    jacobian = - m**3. * c**3. * (nu_c / nu) * gamma**2. / (p_perp * (1. - np.cos(theta) * p_bar))
    
    #dp_perp * dp_parallel from equation 
    dp_perp_dp_para = jacobian #*dadp_bar, which is notation we don't use for a numerical integral
    
    #d^3p from equation 26
    d3p = 2. * np.pi * p_perp * dp_perp_dp_para
    
    ans = M * N / Omega * Df(gamma) * bessel_term * d3p
    
    return ans

# 1/z term integrand
def integrand_other_term(a, p_bar, nu):
    #minimum value of a to keep p_perp real; equation 27
    a_min = (nu / nu_c) * (1. - np.cos(theta) * p_bar) / np.sqrt(1. - p_bar**2.)    
      
    #check if a > a_min
    if(a <= a_min):
        return 0.  
    
    #gamma in terms of a, p_bar; equation 23
    gamma = (nu_c / nu) * a / (1. - np.cos(theta) * p_bar)
    
    #p_parallel, equation 21
    p_parallel = gamma * m * c * p_bar

    #p_perp, equation 22
    p_perp = m * c * np.sqrt(gamma**2. * (1. - p_bar**2.) - 1.)
    
    #this is gamma as a function of p_perp and p_parallel, equation 28
    gamma_check = np.sqrt((p_perp/(m * c))**2. + (p_parallel/(m * c))**2. + 1.)
    
    #Omega, equation 8
    Omega = omega_c / gamma
    
    #z, equation 5
    z = omega * np.sin(theta) * p_perp / (gamma * m * c * Omega) 

    #M, equation 6
    M = (np.cos(theta) - (p_parallel / (gamma * m * c))) / np.sin(theta)

    #N, equation 7
    N = p_perp / (gamma * m * c)
 
    jacobian = - m**3. * c**3. * (nu_c / nu) * gamma**2. / (p_perp * (1. - np.cos(theta) * p_bar))
    
    #dp_perp * dp_0.00740740740741parallel from equation 
    dp_perp_dp_para = jacobian #*dadp_bar, which is notation we don't use for a numerical integral
    
    #d^3p from equation 26
    d3p = 2. * np.pi * p_perp * dp_perp_dp_para
    
    ans = M * N / Omega * Df(gamma) * (1./z) * d3p
    
    return ans

# bessel function term integrand with 1/sin(pi * a) term
def integrand_poles(a, p_bar, nu):
    ans = integrand_nopoles(a, p_bar, nu) / np.sin(np.pi * a)
    return ans

# minimum value of a allowed to keep p_perp real; equation 27
def a_min(p_bar, nu):
    ans = (nu / nu_c) * (1. - np.cos(theta) * p_bar) / np.sqrt(1. - p_bar**2.)    
    return ans

def approx_sign(a):
    n = np.round(a)
    ans = (-1.)**(n % 2)
    return ans

def a_integrator(p_bar, nu):
    
    #for p_bar = -1, 1, a_min goes to infinity
    if(np.isinf(a_min(p_bar, nu)) == True):
        return 0.
    
    #width of range around poles for CPV integrator
    width = 0.01
    
    #starting point for first QUAD integral; should start after pole
    start = np.int(a_min(p_bar, nu) + 1.) + width
    #ending point for first QUAD integral; should stop short of pole
    end   = start + (1. - 2. * width)
    
    #index used to slide integrators along a
    i     = 0.
    
    #declare answer, integral contribution, initial contribution to be 0
    ans                  = 0.
    contrib              = 0.
    initial_contribution = 0.
    
    #integrator quits once the absolute value of the most recent 
    # contribution to the integral divided by the total answer 
    # is smaller than tolerance
    tolerance = 1e-5
    
    #CASE 1: a_min < start - 2 * width; we have to integrate a region without
    #        poles from a_min to start - 2 * width and then integrate the pole
    #        -containing region from start - 2 * width to start + 2 * width.
    if(a_min(p_bar, nu) < start - 2. * width):
        initial_nopoles = quad(lambda a: integrand_poles(a, p_bar, nu), a_min(p_bar, nu), start - 2. * width)
        initial_poles   = quad(lambda a: integrand_nopoles(a, p_bar, nu)/np.pi * approx_sign(a), 
                               start - 2. * width, start, weight='cauchy', wvar=np.round(start))
        initial_contribution = initial_nopoles[0] + initial_poles[0]
           
        #CASE 1a: if both initial_poles and initial_nopoles are 0, ans = their sum
        #         will be zero before the while loop below runs.  The condition in
        #         the loop will then be 0/0 and will throw an error.  This sets ans
        #         to a small nonzero value (negligible since the integral is on the
        #         order of 1e20) so the loop condition does not throw the error.
        if(initial_nopoles[0] == 0 and initial_poles[0] == 0):
            ans = 1e-15

    #CASE 2: a_min > start - 2 * width; we have to integrate a region with a pole
    #        first from a_min to start.
    if(a_min(p_bar, nu) > start - 2. * width):
        initial_poles = quad(lambda a: integrand_nopoles(a, p_bar, nu)/np.pi * approx_sign(a), 
                             a_min(p_bar, nu), start, weight='cauchy', wvar=np.round(a_min(p_bar, nu)))
        
        #CASE 2a: if initial_poles is 0, ans = initial_poles = 0 and the while loop below
        #         throws a divide-by-zero error.  Set ans to a negligible nonzero value
        #         in this case just so that the error does not occur.
        if(initial_poles[0] == 0):
            ans = 1e-15 
                
        initial_contribution = initial_poles[0]
            
    while(i == 0 or np.abs(contrib/ans) > tolerance):  
        between_poles = quad(lambda a: integrand_poles(a, p_bar, nu), start + i, end + i)
        poles         = quad(lambda a: integrand_nopoles(a, p_bar, nu)/np.pi * approx_sign(a), 
                             end + i, end + 2. * width + i, weight='cauchy', wvar=np.round(end + width + i))
        i = i + 1.
                           
        contrib = between_poles[0] + poles[0]
        
        ans = ans + contrib
        
    return ans + initial_contribution

# a integral for 1/z term integrand
def other_term_a_integrator(p_bar, nu):
       
    integral_ans = quad(lambda a: integrand_other_term(a, p_bar, nu), a_min(p_bar, nu), np.inf)
    
    return integral_ans[0]

#FINAL ANSWER (without constant prefactor)
def final_ans(nu):
    order = 35

    #prefactor = ((2. * np.pi * nu)**2. * c**2.) / (4. * np.pi) * e**2. * (2./nu) * m
    prefactor = 1.

    answer_bessel = np.vectorize(fixed_quad)(lambda p_bar: np.vectorize(a_integrator)(p_bar, nu), 
                                             -1., 1., n=order)
    answer_invz   = np.vectorize(fixed_quad)(lambda p_bar: np.vectorize(other_term_a_integrator)(p_bar, nu), 
                                             -1., 1., n=order)
    final_answer  = answer_bessel[0] + answer_invz[0]

#    print 'Bessel func. term:          ', prefactor * answer_bessel[0]
#    print '1/z term:                   ', prefactor * answer_invz[0]
#    print 'alpha_12 final answer:      ', prefactor * final_answer
    return final_answer

print '--------------------OUTPUT--------------------'
print 'nu/nu_c	nu/nu_c_dexter	ans'
print nu/nu_c, nu/nu_c_dexter, final_ans(nu)
