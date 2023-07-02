import numpy as np
import math
from numpy import sin,cos,abs,log,exp
from numpy.linalg import norm
from scipy.ndimage import map_coordinates
import scipy.ndimage
from dataclasses import dataclass
from collections import namedtuple

# Physics Facts

gravity_constant=9.8 #m/s^2
gas_constant=8.3145 #J/(mol K)

sea_level_pressure=101325 #Pa
sea_level_temperature=288.16 #K
temperature_lapse_clamp=215 #K
air_molar_mass=0.02897 #kg/mol
air_heat_capacity_isobaric=1012 #J/(kg K)
air_heat_capacity_isochoric=air_heat_capacity_isobaric-gas_constant/air_molar_mass # note that it's per kg not per mol!
air_heat_capacity_ratio=air_heat_capacity_isobaric/air_heat_capacity_isochoric  #γ=cP/cV

water_molar_mass=0.01802 #kg/mol
water_vaporization_heat_isochoric=2264705 #J/kg
water_melting_heat=334000 #J/kg
rain_droplet_formation_a=1e-3 #kg/kg
rain_droplet_formation_k1=1e-3 #1/s
rain_droplet_formation_k2=2.2 #1/s
rain_droplet_terminal_velocity=10 #m/s


# definition of dimensionless quantities:
# θ=(p0/p)^((γ-1)/γ)T
# π=(p/p0)^((γ-1)/γ)
# θ=T/π
# Q mixing ratio water_mass/air_mass kg/kg 
# since the vapor expand with air, so the ratio is constant alone material derivative



def update_cloud_model(theta,Qv,Qc,Qr,pbar,timeStep):
    'theta=potential temperature at 1atm in K, pbar in Pa, Qvcr in kg/kg'
    T=get_temperature(potential_temperature=theta,pressure=pbar)
    pai=get_dimensionless_pressure(pressure=pbar)
    Qvs=get_saturated_vapor_fraction(temperature=T,pressure=pbar)

    # rain droplets forms by collisions between cloud droplets with cloud droplets or rain droplets
    # dQrf=get_rain_droplet_formation_rate(Qc=Qc,Qr=Qr)*timeStep
    # Qc=Qc-dQrf
    # Qr=Qr+dQrf

    # formation of rain droplet
    dQrf=np.minimum(Qc,get_rain_droplet_formation_rate(Qc=Qc,Qr=Qr)*timeStep)
    Qr=Qr+dQrf
    Qc=np.maximum(0,Qc-dQrf)
    
    # evaporation of rain droplet
    dQre=np.minimum(Qr,get_rain_droplet_evaporation_rate(Qv=Qv,Qr=Qr,T=T,p=pbar)*timeStep)
    Qr=np.maximum(0,Qr-dQre)
    Qv=Qv+dQre

    # if the air is not saturated, cloud droplets vaporize immediately
    # elif the air is oversaturated, vapor condense into cloud droplets
    
    dQce=np.minimum(Qv+Qc,Qvs)-Qv
    Qv=np.maximum(0,Qv+dQce)
    Qc=np.maximum(0,Qc-dQce)

    # heat absorption from vaporization: dE/m_air= -L dQv
    # temperature change: dT=dE/(cp m_air)
    # potential temperature: dθ=dT/π
    # in conclusion, dθ= -L dQv/(π cp)
    # L: J/kg, dQv: kg/kg, cp: J/(kg K), dθ: K
    dtheta=-get_water_vaporization_heat_isobaric(T)/air_heat_capacity_isobaric*(dQce+dQre)/pai

    theta=theta+dtheta

    return theta,Qv,Qc,Qr
    

@dataclass
class Atmosphere_Model_Desc:
    T0:float=sea_level_temperature
    p0:float=sea_level_pressure
    L0:float=gravity_constant/air_heat_capacity_isobaric # 0.00968 #K/m
    h1:float=8000
    L1:float=0
    T_thermal:float=sea_level_temperature+1
    RH_thermal:float=0.3


def atmosphere_model(heights,desc:Atmosphere_Model_Desc):
    T1=desc.T0-desc.h1*desc.L0
    p1=integrate_pressure_at_height(heights_rel=desc.h1,p0=desc.p0,T0=desc.T0,L=desc.L0)
    temperature=np.where(heights<desc.h1,
        desc.T0-desc.L0*(heights),
        T1-desc.L1*(heights-desc.h1))
    pressure=np.where(heights<desc.h1,
        integrate_pressure_at_height(heights_rel=heights,p0=desc.p0,T0=desc.T0,L=desc.L0),
        integrate_pressure_at_height(heights_rel=heights-desc.h1,p0=p1,T0=T1,L=desc.L1))
    density_dry=get_density(temperature=temperature,pressure=pressure)
    return pressure,density_dry,temperature



def integrate_pressure_at_height(heights_rel,p0,T0,L):
    # dp = - ρ g dh, ρ = Mp/RT, T=T0-L h
    # => p ∝ (T/T0)^(Mg/RL)
    # or p ∝ exp(-Mgh/RT) when L=0
    if abs(L)<1e-15:
        index=air_molar_mass*gravity_constant/(gas_constant*T0)
        return p0*exp(-index*heights_rel)
    else:
        index=air_molar_mass*gravity_constant/(gas_constant*L)
        return p0*((T0-L*heights_rel)/T0)**index


def get_density(temperature,pressure):
    # ρ=pM/RT
    return pressure*air_molar_mass/gas_constant/temperature


def get_wet_density(temperature,pressure,Qv=0,Qc=0,Qr=0):
    # Mwet=(1+Qv+Qc+Qr)(1/Mair+Qv/Mwater)
    # ρ=pM/RT
    Qv_coeff=(air_molar_mass-water_molar_mass)/water_molar_mass
    Q_coeff=(1-Qv_coeff*Qv+Qc+Qr)
    density_wet=pressure*air_molar_mass*Q_coeff/gas_constant/temperature
    return density_wet


def get_buoyancy_acceleration(T_ratio,Qv=0,Qc=0,Qr=0):
    'theta in K, Q in kg/kg'
    # Mwet=(1+Qv+Qc+Qr)(1/Mair+Qv/Mwater)
    # ρ=pM/RT
    # T ∝ θ
    # B=-g (ρ/ρbar -1)
    #  = g ((T-Tbar)/Tbar+(Ma-Mw)/Mw Qv - Qc - Qr)
    if np.isscalar(Qv): Qv=np.ones_like(T_ratio)*Qv
    if np.isscalar(Qc): Qc=np.ones_like(T_ratio)*Qc
    if np.isscalar(Qr): Qr=np.ones_like(T_ratio)*Qr
    T_term=T_ratio-1
    Qv_coeff=(air_molar_mass-water_molar_mass)/water_molar_mass
    return gravity_constant*(T_term+Qv_coeff*Qv-Qc-Qr)


def get_rain_droplet_formation_rate(Qc,Qr):
    'Qvcr in kg/kg, timeStep in s, rtval in kg/(kg s)'
    # rain droplets forms by collisions between cloud droplets with cloud droplets or rain droplets
    # dQr/dt=k1(Qc-a)+k2 Qc Qr^0.875
    rate=rain_droplet_formation_k1*np.maximum(0,Qc-rain_droplet_formation_a)+rain_droplet_formation_k2 * Qc * Qr**.875
    return rate
    

def _get_rain_droplet_terminal_velocity_dont_use(Qr, rhobar):
    'rhobar in kg/m^3, Qr in kg/kg, rtval in m/s'
    # RI=360 rhobar Vr Qr. RI in cm/hr
    # Qr=72e-7 (RI)^0.88/rhobar
    # TODO the three formulars are not compatible
    # TODO don't use this!!!
    Vr=3634*np.maximum(0,rhobar*Qr)**0.1364
    return Vr
    

def get_rain_droplet_evaporation_rate(Qv,Qr,T,p):
    'Q in kg/kg, rhobar in kg/m^3, T in K, rtval in kg/(kg s)?'
    # 1mb=100pa
    es_mb=get_water_surface_vapour_pressure(T)/100
    rhobar=get_density(temperature=T,pressure=p)
    Vr=_get_rain_droplet_terminal_velocity_dont_use(Qr=Qr, rhobar=rhobar)
    C=1.6+0.57e-3*Vr**1.5 #ventilation coefficient
    Qvs=get_saturated_vapor_fraction(temperature=T,pressure=p)
    rate=-((Qv/Qvs-1)*C*(rhobar*Qr)**0.525)/(rhobar*(5.4e5+0.41e7/es_mb))
    return rate

def get_water_vaporization_heat_isobaric(T):
    return water_vaporization_heat_isochoric-gas_constant*T/water_molar_mass


def get_water_surface_vapour_pressure(T):
    'T in K, rtval in Pa'
    # 1mmHg=133.3224 Pa
    P=133.3224*np.exp(20.386-5132/T)
    return P

def get_relative_humidity(temperature,pressure,Qv):
    return Qv/get_saturated_vapor_fraction(temperature=temperature,pressure=pressure)

def get_saturated_vapor_fraction(temperature,pressure):
    pvs=get_saturated_vapor_pressure(temperature=temperature)
    Qvs=water_molar_mass/air_molar_mass*pvs/pressure
    return Qvs


def get_saturated_vapor_pressure(temperature):
    'temperature in K, rtval in Pa'
    T=temperature
    return np.where(T>273.15,
        610.78*exp(17.27*(T-273.15)/(T-35.85)),
        610.78*exp(21.875*(T-273.15)/(T-7.65))
        )


def get_potential_temperature(temperature,pressure):
    'pressure in Pa, temperature in K, rtval in K, reference pressure is 1atm'
    # θ=Temperature after bring to sea level adiabatically
    # PV^γ=constant
    # P^(1-γ)T^γ=constant
    # T ∝ P^((γ-1)/γ) =: πθ
    # θ=(p0/p)^((γ-1)/γ)T
    return temperature/get_dimensionless_pressure(pressure)


def get_temperature(potential_temperature,pressure):
    'pressure in Pa, potential_temperature in K, rtval in K'
    # T=T0 (p/p0)^((γ-1)/γ)θ
    return potential_temperature*get_dimensionless_pressure(pressure)



def get_dimensionless_pressure(pressure):
    'pressure in Pa, reference pressure is 1atm'
    # π=(p/p0)^((γ-1)/γ)
    index=(air_heat_capacity_ratio-1)/air_heat_capacity_ratio
    return (pressure/sea_level_pressure)**index



def calc_total_thermal_energy(rho,theta,pbar):
    T=get_temperature(potential_temperature=theta,pressure=pbar)
    dry_air_energy=np.sum(rho*T*air_heat_capacity_isochoric)
    assert False
    return dry_air_energy
    

def calc_total_water(Qv,Qc,Qr,rhobar,cellVolumes):
    mv=np.sum(Qv*rhobar*cellVolumes)
    mc=np.sum(Qc*rhobar*cellVolumes)
    mr=np.sum(Qr*rhobar*cellVolumes)
    return mv,mc,mr

# ========== display ==========



def show_buoyancy_analysis(heights,T0,RH0,Tbar,pbar):
    from matplotlib import pyplot as plt

    rhobar=get_density(temperature=Tbar,pressure=pbar)
    T=get_temperature(potential_temperature=T0,pressure=pbar)
    Qvs=get_saturated_vapor_fraction(temperature=T,pressure=pbar)
    Qvs0=get_saturated_vapor_fraction(temperature=T0,pressure=sea_level_pressure)
    Qv0=Qvs0*RH0
    dQc=np.maximum(Qv0-Qvs,0)
    T=T+dQc*get_water_vaporization_heat_isobaric(T)/air_heat_capacity_isobaric
    rho=get_wet_density(temperature=T,pressure=pbar,Qv=Qv0)
    B=get_buoyancy_acceleration(T_ratio=T/Tbar,Qv=Qv0,Qc=0,Qr=0)

    plt.figure(figsize=(12,3))

    plt.subplot(141).plot(Tbar-273.15,heights/1000,color='grey',label='ambient')
    plt.subplot(141).plot(T-273.15,heights/1000,color='red',label='thermal')
    plt.legend()
    plt.grid()
    plt.ylabel('height/km')
    plt.xlabel('temperature/C')

    # plt.subplot(152).plot(pbar,heights/1000,color='grey',label='ambient')
    # plt.legend()
    # plt.grid()
    # plt.ylabel('height/km')
    # plt.xlabel('pressure/Pa')

    plt.subplot(142).plot(rhobar,heights/1000,color='grey',label='ambient')
    plt.subplot(142).plot(rho,heights/1000,color='red',label='thermal')
    plt.legend()
    plt.grid()
    plt.ylabel('height/km')
    plt.xlabel('density/(kg/m^3)')

    plt.subplot(143).plot(B,heights/1000,color='red')
    plt.axvline(0,color='black')
    plt.grid()
    plt.xlim([-.2,.2])
    plt.ylabel('height/km')
    plt.xlabel('buoyancy acceleration/(m/s^2)')

    plt.subplot(144).plot(Qvs*1000,heights/1000,color='grey',label='saturated')
    plt.subplot(144).plot(Qv0*1000*np.ones_like(heights),heights/1000,color='red',label='thermal')
    plt.grid()
    plt.legend()
    plt.ylabel('height/km')
    plt.xlabel('vapor fraction/(g/kg)')

    plt.tight_layout()
    plt.show()

    
def show_rain_models():
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12,3))

    # T=np.linspace(273.15-20,273.15+120)
    # plt.subplot(141).plot(T-273.15,get_water_surface_vapour_pressure(T))
    # plt.axhline(sea_level_pressure,color='black')
    # plt.axvline(100,color='black')
    # plt.grid()
    # plt.xlabel('temperature/C')
    # plt.ylabel('Water Surface Vapour Pressure/Pa')

    Qr=np.linspace(0,1e-4)
    rhobar=get_density(sea_level_pressure,sea_level_temperature)
    plt.subplot(141).plot(Qr*1000,_get_rain_droplet_terminal_velocity_dont_use(Qr,rhobar))
    plt.xlabel('Rain Mixing Ratio Qr/(g/kg)')
    plt.ylabel('Rain Droplet Terminal Velocity/(m/s)')

    Qr=np.linspace(0,1e-4)
    Qc=np.linspace(0,4e-3)
    Qr,Qc=np.meshgrid(Qr,Qc)
    Qrf=get_rain_droplet_formation_rate(Qc=Qc,Qr=Qr)
    contour=plt.subplot(142).contour(Qr*1000,Qc*1000,Qrf*1000,linewidths=1)
    plt.gca().clabel(contour, contour.levels,fmt=lambda x:f'{x:.4f}g/kg/s')
    plt.xlabel('Rain Mixing Ratio Qr/(g/kg)')
    plt.ylabel('Cloud Mixing Ratio Qc/(g/kg)')
    plt.title('Rain Droplet Formation Rate')

    Qr=np.linspace(0,1e-4)
    Qv=np.linspace(0,4e-3)
    Qr,Qv=np.meshgrid(Qr,Qc)
    Qre=get_rain_droplet_evaporation_rate(Qv=Qv,Qr=Qr,T=sea_level_temperature,p=sea_level_pressure)
    contour=plt.subplot(143).contour(Qr*1000,Qv*1000,Qre*1000,linewidths=1)
    plt.gca().clabel(contour, contour.levels,fmt=lambda x:f'{x:.5f}g/kg/s')
    plt.xlabel('Rain Mixing Ratio Qr/(g/kg)')
    plt.ylabel('Vapor Mixing Ratio Qc/(g/kg)')
    plt.title('Rain Droplet Evaporation Rate')

    Qr=np.linspace(0,1e-4)
    h=np.linspace(0,10000)
    Qr,h=np.meshgrid(Qr,h)
    p,rho,T=atmosphere_model(heights=h,desc=Atmosphere_Model_Desc())
    Qre=get_rain_droplet_evaporation_rate(Qv=0,Qr=Qr,T=T,p=p)
    contour=plt.subplot(144).contour(Qr*1000,h,Qre*1000,linewidths=1)
    plt.gca().clabel(contour, contour.levels,fmt=lambda x:f'{x:.5f}g/kg/s')
    plt.xlabel('Rain Mixing Ratio Qr/(g/kg)')
    plt.ylabel('height/m')
    plt.title('Rain Droplet Evaporation Rate')



    plt.tight_layout()
    plt.show()

