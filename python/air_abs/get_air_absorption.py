##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: get_air_absorption.py
#
# Description: This is an implementation of formulae in the ISO9613-1 standard
# for air absorption
#
##############################################################################

import numpy as np
from numpy import array as npa
from numpy import log10, exp, sqrt, log, pi

def get_air_absorption(freq_vec,temperature_celsius,rel_humidity_pnct,pressure_atmospheric_kPa=101.325):
    assert pressure_atmospheric_kPa<=200
    assert temperature_celsius >= -20
    assert temperature_celsius <= 50
    assert rel_humidity_pnct <= 100
    assert rel_humidity_pnct >= 10

    f = freq_vec
    T = temperature_celsius
    rh = rel_humidity_pnct

    f2=f*f
    pi2 = pi*pi

    #convert temperature to kelvin
    Tk=T+273.15

    #triple point isothermal temperature (Section B)
    T01=273.16 #kelvin

    #standard temperature
    T0=293.15

    #ambient pressure
    pa=101.325 #kPa
    #reference pressure (standard)
    pr=101.325 #kPa

    #characteristic vibrational temperature (A.8)
    thO=2239.1
    thN=3352.0
    #fractional molar concentrations (A.8)
    XO=0.209
    XN=0.781

    #constant (A.9)
    const=2*pi/35*(10*log10(exp(2))) #1.559

    #(A.6)
    almO=const*XO*(thO/Tk)**2*exp(-thO/Tk)
    #(A.7)
    almN=const*XN*(thN/Tk)**2*exp(-thN/Tk)

    #pressure ratio
    p=pa/pr
    #temperature ratio
    Tr=Tk/T0

    #speed of sound
    c=343.2*sqrt(Tr)
    c2 = c*c

    #(B.3)
    C=-6.8346*(T01/Tk)**1.261 + 4.6151
    #(B.1) and (B.2)
    h=rh*(10**C)*p 

    #relaxation frequencies
    #(3)
    frO = p* (24 + 4.04e4 * h * (0.02 + h)/(0.391 + h))

    #(4)
    frN = p * Tr**(-0.5) * (9 + 280 * h * exp(-4.17 * (Tr**(-1/3) - 1)))

    #(5) 
    absfull1=8.686*f2*(1.84e-11 * sqrt(Tr)/p + Tr**-2.5 * (0.01275*(exp(-2239.1/Tk)/(frO + f2/frO)) + 0.1068*(exp(-3352.0/Tk)/(frN + f2/frN))))

    #(A.2)
    absClRo=1.6e-10*sqrt(Tr)*f2/p

    #derived 
    eta = log(10)*1.6e-11/(4*pi2)*(c2)*sqrt(Tr)/p

    #(A.3)
    absVibO=almO*(f/c)*(2*(f/frO)/(1+(f/frO)**2))
    #(A.4)
    absVibN=almN*(f/c)*(2*(f/frN)/(1+(f/frN)**2))
    #(A.1)
    absfull2 = absClRo + absVibO + absVibN

    assert np.allclose(absfull1,absfull2,rtol=1e-2)

    #modified viscothermal coefficient (see FA2014 or DAFx2021 papers)
    etaO = almO*(c/pi2/frO)*log(10)/20

    #return a dictionary of different constants
    return_dict = {}
    return_dict['gamma_p'] = etaO/c
    return_dict['gamma'] = eta/c
    return_dict['etaO'] = etaO
    return_dict['eta'] = eta
    return_dict['almN'] = almN
    return_dict['almO'] = almO
    return_dict['c'] = c
    return_dict['frO'] = frO
    return_dict['frN'] = frN

    #frequency-dependent coefficeints in Np/m or dB/m
    return_dict['absVibN_dB'] = absVibN
    return_dict['absVibO_dB'] = absVibO
    return_dict['absClRo_dB'] = absClRo
    return_dict['absfull_dB'] = absfull2
    return_dict['absVibN_Np'] = absVibN*log(10)/20
    return_dict['absVibO_Np'] = absVibO*log(10)/20
    return_dict['absClRo_Np'] = absClRo*log(10)/20
    return_dict['absfull_Np'] = absfull2*log(10)/20

    return return_dict

def main():
    from numpy.random import random_sample
    
    f = np.logspace(log10(1),log10(80e3))
    rh = 15
    Tc = 10
    print(f'{Tc=} {rh=}%')

    rd = get_air_absorption(f,Tc,rh)
    print(f"{rd['almO']=}")
    print(f"{rd['almN']=}")
    print(f"{rd['c']=}")
    print(f"{rd['frO']=}")
    print(f"{rd['frN']=}")
    print(f"{rd['eta']=}")

    rh = (100 - 10)*random_sample()+10
    Tc = (50 - -20)*random_sample()-20
    print(f'{Tc=} {rh=}%')
    rd = get_air_absorption(f,Tc,rh)
    print(f"{rd['almO']=}")
    print(f"{rd['almN']=}")
    print(f"{rd['c']=}")
    print(f"{rd['frO']=}")
    print(f"{rd['frN']=}")
    print(f"{rd['eta']=}")

if __name__ == '__main__':
    main()
