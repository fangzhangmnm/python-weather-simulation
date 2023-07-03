import numpy as np
import math
from numpy import sin,cos,abs
from numpy.linalg import norm
from scipy.ndimage import map_coordinates
import scipy.ndimage
from dataclasses import dataclass
from typing import Literal

# https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/stable_fluids_python_simple.py

# (∂/∂t + u ⋅ ∇) (ρu) = − ∇p + ν ρ ∇²u + f

cellSizes=np.array((1/128,1/128))
timeStep=0.01
gridRes=np.array((128,128))

Boundary_Condition=Literal['open','fixed','periodic'] #open is not goot at the moment
boundary_xp:Boundary_Condition='fixed'
boundary_xm:Boundary_Condition='fixed'
boundary_yp:Boundary_Condition='fixed'
boundary_ym:Boundary_Condition='fixed'

def advect(field,velocity):
    # φ' = φ - (u ⋅ ∇)φ dt
    positions=get_positions()
    old_positions=bound_positions_(positions-velocity*timeStep)
    return sample_indices(field,positions_to_indices(old_positions))

advect_vector=advect # be catious of geometric terms when using curved coordinates

def advect_velocity_MacCormack(velocity):
    v0=velocity
    v1hat=advect_vector(v0,v0)
    v0hat=advect_vector(v1hat,-v1hat)
    v1=v1hat+.5*(v0-v0hat)
    v1=np.clip(v1,np.minimum(v0,v1hat),np.maximum(v0,v1hat)) #Limiter
    return v1

def solve_pressure(velocity,density,nIter,old_pressure=None):
    # u" = u' − 1/ρ ∇p dt
    # ∇²p = ∇ ⋅ (ρ u' / dt)
    if isinstance(density,np.ndarray):
        density=density[...,None]
    RHS=divergence(density/timeStep * velocity)
    if old_pressure is not None:
        old_pressure=old_pressure-np.average(old_pressure,axis=None)
    pressure=inverse_laplacian(RHS,nIter=nIter,initial_guess=old_pressure,
        bc_fn_=apply_neumann_boundary_)
    projected_velocity=velocity-timeStep/density*grad(pressure)
    return projected_velocity,pressure
    
# pressure_correction_K=1
# def solve_pressure_fast(u,p):
#     if p is None: p=np.ones(u.shape[:-1])
#     #p=advect(p,u)
#     gradp=grad(p)
#     p=p-timeStep*(divergence(u)*p+(gradp*u).sum(-1))
#     p=np.clip(p,.5,3)
#     u=u-pressure_correction_K*timeStep*gradp
#     return u,p
    
   
def diffuse(field,amount,nIter):
    # WARNING! here we omit geometric term!!!
    # (1-ν dt ∇²) u'= u
    #               = (1 + 4ν dt/dx**2) diag + ( -ν dt/dx**2) off_diag
    rtval=field.copy()
    if amount!=0:
        offDiagCoeffs=-amount/np.asarray(cellSizes)**2
        diagCoeff=1-2*np.sum(offDiagCoeffs)
        for _iter in range(nIter):
            rtval=jacobi_iteration_step(x=rtval,b=field,diagCoeff=diagCoeff,offDiagCoeffs=offDiagCoeffs)
            rtval=apply_dirichlet_boundary_(rtval,field)
    return rtval

def amplify_vorticity(velocity,vorticity_eps):
    # ω=▽×u
    # u=u+eps dt dx normalize(▽|ω|)×ω
    vorticity=curl(velocity)
    if vorticity_eps!=0:
        eta=grad(abs(vorticity))
        eta=eta/np.maximum(norm(eta,axis=-1)[...,None],1e-7)
        eta=np.stack((eta[...,1],-eta[...,0]),axis=-1)*vorticity[...,None]
        velocity=velocity+vorticity_eps*timeStep*np.min(cellSizes)*eta
    return velocity,vorticity

def add_buoyancy(velocity,acceleration):
    # ∇pbar=ρbar g
    # g-∇p/ρ=-∇(p-pbar)/ρbar+g(ρ-ρbar)/ρbar
    # (ρ-ρbar)/ρbar=-(T-Tbar)/Tbar
    velocity=velocity.copy()
    velocity[...,-1]+=timeStep*acceleration# gravity*(temperature_ratio-1)
    return velocity

def calc_total_kinematic_energy(u,rho):
    return .5*integrate(rho*np.sum(u**2,axis=-1))


# ========== Examples ==========

def taylor_green_vortex():
    uv=get_normalized_positions()
    u0=np.zeros_like(uv)
    u0[...,0]=np.cos(uv[...,0]*np.pi*2)*np.sin(uv[...,1]*np.pi*2)
    u0[...,1]=-np.sin(uv[...,0]*np.pi*2)*np.cos(uv[...,1]*np.pi*2)
    return u0

def laminar_flow():
    uv=get_normalized_positions()
    u0=np.zeros_like(uv)
    u0[...,0]=np.cos(uv[...,-1]*np.pi*2+np.pi)
    return u0

def diffused_noise(shape,diffuse_amount,nIter):
    rtval=np.random.randn(*shape)
    rtval=diffuse(rtval,amount=diffuse_amount,nIter=nIter)
    return rtval

# ========== Vector Calculus ==========

def integrate(field):
    return np.sum(field,axis=(0,1))*np.prod(cellSizes)

def grad_x(field):
    return (roll_field(field,-1,axis=0)-roll_field(field,1,axis=0))/(2*cellSizes[0])


def grad_y(field):
    return (roll_field(field,-1,axis=1)-roll_field(field,1,axis=1))/(2*cellSizes[1])


def divergence(vector_field):
    return grad_x(vector_field[...,0])+grad_y(vector_field[...,1])


def grad(field):
    return np.stack([grad_x(field),grad_y(field)],axis=-1)


def curl(vector_field):
    return grad_x(vector_field[...,1])-grad_y(vector_field[...,0])


def laplacian(field):
    offDiagCoeff=1/np.asarray(cellSizes)**2
    diagCoeff=-4*offDiagCoeff
    rtval=diagCoeff*field.copy()
    rtval+=offDiagCoeff[0]*(roll_field(field,-1,axis=0)+roll_field(field,1,axis=0))
    rtval+=offDiagCoeff[1]*(roll_field(field,-1,axis=1)+roll_field(field,1,axis=1))
    return rtval


def inverse_laplacian(field,nIter,initial_guess=None,bc_fn_=None):
    # ∇² x = b
    #      = (-4/dx**2) diag + (1/dx**2) off_diag
    offDiagCoeffs=1/np.asarray(cellSizes)**2
    diagCoeff=-2*np.sum(offDiagCoeffs)
    rtval=np.zeros_like(field) if initial_guess is None else initial_guess.copy()
    for _iter in range(nIter):
        rtval=jacobi_iteration_step(x=rtval,b=field,diagCoeff=diagCoeff,offDiagCoeffs=offDiagCoeffs)
        if bc_fn_:
            rtval=bc_fn_(rtval)
    return rtval





def jacobi_iteration_step(x,b,diagCoeff,offDiagCoeffs):
    # (Σ + L) x = b
    # x' = b/Σ - L/Σ x
    # alpha = coeff of 1/Σ
    # beta = coeff of -L/Σ
    alpha=1/diagCoeff
    betas=-offDiagCoeffs/diagCoeff
    xnew=alpha*b
    xnew+=betas[0]*(roll_field(x,-1,axis=0)+roll_field(x,1,axis=0))
    xnew+=betas[1]*(roll_field(x,-1,axis=1)+roll_field(x,1,axis=1))
    return xnew

# ========== Boundary Conditions ==========





def apply_neumann_boundary_(field):
    if boundary_xm=='fixed':
        field[0,...]=field[1,...]
    if boundary_xp=='fixed':
        field[-1,...]=field[-2,...]
    if boundary_ym=='fixed':
        field[:,0,...]=field[:,1,...]
    if boundary_yp=='fixed':
        field[:,-1,...]=field[:,-2,...]
    return field


def apply_dirichlet_boundary_(field,field0):
    if boundary_xm=='fixed':
        field[0,...]=field0[0,...]
    if boundary_xp=='fixed':
        field[-1,...]=field0[-1,...]
    if boundary_ym=='fixed':
        field[:,0,...]=field0[:,0,...]
    if boundary_yp=='fixed':
        field[:,-1,...]=field0[:,-1,...]
    return field


def bound_positions_(positions):
    if boundary_xm=='periodic':
        positions[...,0]%=gridRes[0]*cellSizes[0]
    else:
        positions[...,0]=np.clip(positions[...,0],.5*cellSizes[0],(gridRes[0]-.5)*cellSizes[0])
    if boundary_ym=='periodic':
        positions[...,1]%=gridRes[1]*cellSizes[1]
    else:
        positions[...,1]=np.clip(positions[...,1],.5*cellSizes[1],(gridRes[1]-.5)*cellSizes[1])
    return positions


def roll_field(field,dir,axis):
    assert abs(dir)==1
    rtval=np.roll(field,shift=dir,axis=axis)
    if axis==0 and boundary_xm!='periodic':
        if dir>0:
            rtval[0,...]=field[0,...]
        else:
            rtval[-1,...]=field[-1,...]
    if axis==1 and boundary_ym!='periodic':
        if dir>0:
            rtval[:,0,...]=field[:,0,...]
        else:
            rtval[:,-1,...]=field[:,-1,...]
    return rtval

# ========== Texture Lookup Conventions ==========



def sample_indices(field,indices):
    rtval=np.empty(indices.shape[:2]+(math.prod(field.shape[2:]),))
    field1=field.reshape(field.shape[:2]+(-1,))
    for i in range(rtval.shape[2]):
        rtval[...,i]=map_coordinates(field1[...,i],indices.transpose(2,0,1),order=1,mode='grid-wrap')
    rtval=rtval.reshape(indices.shape[:2]+field.shape[2:])
    return rtval
    

def get_positions():
    #  tex id   |  0  |  1  |  2  |  3  |  4  |
    #  uv       0     1     2     3     4     5
    x=np.linspace(.5,gridRes[0]-.5,gridRes[0])*cellSizes[0]
    y=np.linspace(.5,gridRes[1]-.5,gridRes[1])*cellSizes[1]
    xy=np.meshgrid(x,y,indexing='ij')
    return np.stack(xy,axis=-1)

def positions_to_indices(positions):
    #  tex id   |  0  |  1  |  2  |  3  |  4  |
    #  uv       0     1     2     3     4     5
    return positions/cellSizes-.5

def get_normalized_positions():
    return get_positions()/(np.asarray(gridRes)*cellSizes)
    


# ========== Display ==========
from matplotlib import pyplot as plt

def show_quiver(xy,u,numbers=[10,10],**args):
    zoom=[numbers[0]/xy.shape[0],numbers[1]/xy.shape[1],1]
    xy=scipy.ndimage.zoom(xy,zoom=zoom,order=1)
    u=scipy.ndimage.zoom(u,zoom=zoom,order=1)
    plt.gca().set_aspect('equal')
    return plt.quiver(xy[...,0],xy[...,1],u[...,0],u[...,1],**args)

def show_image(values,mask=None,is_rgb=False,**args):
    if len(values.shape)==3 and not is_rgb:
        values=norm(values,axis=-1)
    if mask is not None:
        args['alpha']=np.where(mask,0.,1.).swapaxes(0,1)*args.get('alpha',1)
        if not isinstance(values,np.ndarray):
            values=np.ones_like(mask)*values
    shape=values.shape[:2]
    extent=[0,shape[0]*cellSizes[0],0,shape[1]*cellSizes[1]]
    return plt.imshow(values.copy().swapaxes(0,1),extent=extent,origin='lower',**args) 


def show_contour(xy,values,fmt=None,**args):
    if len(values.shape)==3:
        values=norm(values,axis=-1)
    x,y=xy[...,0],xy[...,1]
    plt.contourf(x,y,values,cmap='viridis',**args)
    contour=plt.contour(x,y,values,colors='black',linestyles='dashed',linewidths=1,**args)
    plt.gca().clabel(contour, contour.levels,fmt=fmt)
    plt.gca().set_aspect('equal')

