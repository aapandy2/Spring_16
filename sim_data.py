#file to calculate <j_nu()> vs. j_nu(<avgs>)

import symphonyPy as sp
import numpy as np
import pylab as pl

#set constant parameters for the calculation
m = 9.1093826e-28
c = 2.99792458e10
theta_e = 10. #Do we need to do anything about electron temp?
e = 4.80320680e-10
h = 6.6260693e-27
gamma_min = 1.
gamma_max = 1000.
gamma_cutoff = 1e10
power_law_p = 3.5
kappa = 3.5
kappa_width = 10.
nuratio = 1.e2
B_scale = 30. #TODO: check if sim data is actually normalized to init. val.

#import data from Dr. Kunz's simulation
B_x = np.loadtxt('/home/alex/Documents/Spring_2016/mirror_bx.out') * B_scale
B_y = np.loadtxt('/home/alex/Documents/Spring_2016/mirror_by.out') * B_scale
B_z = np.loadtxt('/home/alex/Documents/Spring_2016/mirror_bz.out') * B_scale
B_mag = np.sqrt(B_x**2. + B_y**2. + B_z**2.)
n_e = np.loadtxt('/home/alex/Documents/Spring_2016/mirror_d.out')

#still thinking about how to do observer_angle.  Is there a better way than this?
obs_angle = np.arccos(1./(1.) * (0.*B_x + -1.*B_y + 0.*B_z)/B_mag)

#Generate all averages
x = 0
y = 0
B_x_avg        = 0
B_y_avg        = 0
B_z_avg        = 0
n_e_avg        = 0
obs_angle_avg  = 0

for x in range(0, 1151):
  for y in range(0, 1151):
    B_x_avg       = B_x_avg + B_x[x][y]
    B_y_avg       = B_y_avg + B_y[x][y]
    B_z_avg       = B_z_avg + B_z[x][y]
    n_e_avg       = n_e_avg + n_e[x][y]
    obs_angle_avg = obs_angle_avg + obs_angle[x][y]

B_x_avg        = B_x_avg/(1152*1152)
B_y_avg        = B_y_avg/(1152*1152)
B_z_avg        = B_z_avg/(1152*1152)
n_e_avg        = n_e_avg  /(1152*1152)
B_mag_avg      = np.sqrt(B_x_avg**2. + B_y_avg**2. + B_z_avg**2.)
obs_angle_avg  = obs_angle_avg/(1152*1152) 


#-------------------------------MJ_I-------------------------------------------#
x = 0
y = 0
MJ_I_exact_avg = 0
MJ_I_exact = [[0 for i in range(1152)] for j in range(1152)]

for x in range(0, 1151):
  for y in range(0, 1151):
    MJ_I_exact[x][y] = sp.j_nu_fit_py(nuratio, B_mag[x][y], n_e[x][y], 
                                      obs_angle[x][y], sp.MAXWELL_JUETTNER, 
                                      sp.STOKES_I, theta_e, power_law_p, 
                                      gamma_min, gamma_max, gamma_cutoff, 
                                      kappa, kappa_width)
    MJ_I_exact_avg = MJ_I_exact_avg + MJ_I_exact[x][y]

MJ_I_exact_avg = MJ_I_exact_avg/(1152*1152)

MJ_I_avgs  = sp.j_nu_fit_py(nuratio, B_mag_avg, n_e_avg,
                            obs_angle_avg, sp.MAXWELL_JUETTNER,
                            sp.STOKES_I, theta_e, power_law_p,
                            gamma_min, gamma_max, gamma_cutoff,
                            kappa, kappa_width)

print 'MJ_I', '(<j_nu()>-j_nu(<>))/<j_nu()> = ', (MJ_I_exact_avg 
                                                  - MJ_I_avgs)/MJ_I_exact_avg


#-------------------------------MJ_Q-------------------------------------------#
x = 0
y = 0
MJ_Q_exact_avg = 0
MJ_Q_exact = [[0 for i in range(1152)] for j in range(1152)]

for x in range(0, 1151):
  for y in range(0, 1151):
    MJ_Q_exact[x][y] = sp.j_nu_fit_py(nuratio, B_mag[x][y], n_e[x][y],
                                      obs_angle[x][y], sp.MAXWELL_JUETTNER,
                                      sp.STOKES_Q, theta_e, power_law_p,
                                      gamma_min, gamma_max, gamma_cutoff,
                                      kappa, kappa_width)
    MJ_Q_exact_avg = MJ_Q_exact_avg + MJ_Q_exact[x][y]

MJ_Q_exact_avg = MJ_Q_exact_avg/(1152*1152)

MJ_Q_avgs  = sp.j_nu_fit_py(nuratio, B_mag_avg, n_e_avg,
                            obs_angle_avg, sp.MAXWELL_JUETTNER,
                            sp.STOKES_Q, theta_e, power_law_p,
                            gamma_min, gamma_max, gamma_cutoff,
                            kappa, kappa_width)

print 'MJ_Q', '(<j_nu()>-j_nu(<>))/<j_nu()> = ', (MJ_Q_exact_avg
                                                  - MJ_Q_avgs)/MJ_Q_exact_avg


#-------------------------------MJ_V-------------------------------------------#
x = 0
y = 0
MJ_V_exact_avg = 0
MJ_V_exact = [[0 for i in range(1152)] for j in range(1152)]

for x in range(0, 1151):
  for y in range(0, 1151):
    MJ_V_exact[x][y] = sp.j_nu_fit_py(nuratio, B_mag[x][y], n_e[x][y],
                                      obs_angle[x][y], sp.MAXWELL_JUETTNER,
                                      sp.STOKES_V, theta_e, power_law_p,
                                      gamma_min, gamma_max, gamma_cutoff,
                                      kappa, kappa_width)
    MJ_V_exact_avg = MJ_V_exact_avg + MJ_V_exact[x][y]

MJ_V_exact_avg = MJ_V_exact_avg/(1152*1152)

MJ_V_avgs  = sp.j_nu_fit_py(nuratio, B_mag_avg, n_e_avg,
                            obs_angle_avg, sp.MAXWELL_JUETTNER,
                            sp.STOKES_V, theta_e, power_law_p,
                            gamma_min, gamma_max, gamma_cutoff,
                            kappa, kappa_width)

print 'MJ_V', '(<j_nu()>-j_nu(<>))/<j_nu()> = ', (MJ_V_exact_avg
                                                  - MJ_V_avgs)/MJ_V_exact_avg
