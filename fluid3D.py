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

cellSizes=np.array((1/48,1/48,1/48))
timeStep=0.01
gridRes=np.array((48,48,48))

Boundary_Condition=Literal['open','fixed','periodic'] #open is not goot at the moment
boundary_xp:Boundary_Condition='periodic'
boundary_xm:Boundary_Condition='periodic'
boundary_yp:Boundary_Condition='periodic'
boundary_ym:Boundary_Condition='periodic'
boundary_zp:Boundary_Condition='periodic'
boundary_zm:Boundary_Condition='periodic'

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

def amplify_vorticity(velocity,vorticity_eps):
    # ω=▽×u
    # u=u+eps dt dx normalize(▽|ω|)×ω
    vorticity=curl(velocity)
    if vorticity_eps!=0:
        eta=grad(norm(vorticity,axis=-1))
        eta=eta/np.maximum(norm(eta,axis=-1)[...,None],1e-7)
        eta=np.cross(eta,vorticity)
        velocity=velocity+vorticity_eps*timeStep*np.min(cellSizes)*eta
    return velocity,vorticity

def add_buoyancy(velocity,acceleration):
    # ∇pbar=ρbar g
    # g-∇p/ρ=-∇(p-pbar)/ρbar+g(ρ-ρbar)/ρbar
    # (ρ-ρbar)/ρbar=-(T-Tbar)/Tbar
    velocity=velocity.copy()
    velocity[...,-1]+=timeStep*acceleration# gravity*(temperature_ratio-1)
    return velocity

def add_coriolis_force(velocity,w=np.array([0,0,1.99e-7*.71])):
    return velocity-2*timeStep*np.cross(velocity,w)

def calc_total_kinematic_energy(u,rho):
    return .5*integrate(rho*np.sum(u**2,axis=-1))

# ========== Examples ==========

def taylor_green_vortex(A=1,B=-1,C=0):
    uvw=get_normalized_positions()
    u0=np.zeros_like(uvw)
    u0[...,0]=A*np.cos(uvw[...,0]*np.pi*2)*np.sin(uvw[...,1]*np.pi*2)*np.sin(uvw[...,2]*np.pi*2)
    u0[...,1]=B*np.sin(uvw[...,0]*np.pi*2)*np.cos(uvw[...,1]*np.pi*2)*np.sin(uvw[...,2]*np.pi*2)
    u0[...,2]=C*np.sin(uvw[...,0]*np.pi*2)*np.cos(uvw[...,1]*np.pi*2)*np.sin(uvw[...,2]*np.pi*2)
    return u0

def laminar_flow():
    uvw=get_normalized_positions()
    u0=np.zeros_like(uvw)
    u0[...,0]=np.sin(uvw[...,1]*np.pi)*np.sin(uvw[...,2]*np.pi)
    return u0

def diffused_noise(shape,diffuse_amount=1,nIter=10):
    rtval=np.random.randn(*shape)
    rtval=diffuse(rtval,amount=diffuse_amount,nIter=nIter)
    return rtval

# ========== Vector Calculus ==========

def integrate(field):
    return np.sum(field,axis=(0,1,2))*np.prod(cellSizes)


def grad_x(field):
    return (roll_field(field,-1,axis=0)-roll_field(field,1,axis=0))/(2*cellSizes[0])


def grad_y(field):
    return (roll_field(field,-1,axis=1)-roll_field(field,1,axis=1))/(2*cellSizes[1])


def grad_z(field):
    return (roll_field(field,-1,axis=2)-roll_field(field,1,axis=2))/(2*cellSizes[2])

def grad(field):
    return np.stack([grad_x(field),grad_y(field),grad_z(field)],axis=-1)

def divergence(vector_field):
    return grad_x(vector_field[...,0])+grad_y(vector_field[...,1])+grad_z(vector_field[...,2])


def curl(vector_field):
    return np.stack([
        grad_y(vector_field[...,2])-grad_z(vector_field[...,1]),
        grad_z(vector_field[...,0])-grad_x(vector_field[...,2]),
        grad_x(vector_field[...,1])-grad_y(vector_field[...,0]),
    ],axis=-1)


def laplacian(field):
    offDiagCoeff=1/np.asarray(cellSizes)**2
    diagCoeff=-6*offDiagCoeff
    rtval=diagCoeff*field.copy()
    rtval+=offDiagCoeff[0]*(roll_field(field,-1,axis=0)+roll_field(field,1,axis=0))
    rtval+=offDiagCoeff[1]*(roll_field(field,-1,axis=1)+roll_field(field,1,axis=1))
    rtval+=offDiagCoeff[2]*(roll_field(field,-1,axis=2)+roll_field(field,1,axis=2))
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



def diffuse(field,amount,nIter):
    # WARNING! here we omit geometric term!!!
    # (1-ν dt ∇²) u'= u
    #               = (1 + 6ν dt/dx**2) diag + ( -ν dt/dx**2) off_diag
    rtval=field.copy()
    if amount!=0:
        offDiagCoeffs=-amount/np.asarray(cellSizes)**2
        diagCoeff=1-2*np.sum(offDiagCoeffs)
        for _iter in range(nIter):
            rtval=jacobi_iteration_step(x=rtval,b=field,diagCoeff=diagCoeff,offDiagCoeffs=offDiagCoeffs)
            rtval=apply_dirichlet_boundary_(rtval,field)
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
    xnew+=betas[2]*(roll_field(x,-1,axis=2)+roll_field(x,1,axis=2))
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
    if boundary_zm=='fixed':
        field[:,:,0,...]=field[:,:,1,...]
    if boundary_zp=='fixed':
        field[:,:,-1,...]=field[:,:,-2,...]
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
    if boundary_zm=='fixed':
        field[:,:,0,...]=field0[:,:,0,...]
    if boundary_zp=='fixed':
        field[:,:,-1,...]=field0[:,:,-1,...]
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
    if boundary_zm=='periodic':
        positions[...,2]%=gridRes[2]*cellSizes[2]
    else:
        positions[...,2]=np.clip(positions[...,2],.5*cellSizes[2],(gridRes[2]-.5)*cellSizes[2])
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
    if axis==2 and boundary_zm!='periodic':
        if dir>0:
            rtval[:,:,0,...]=field[:,:,0,...]
        else:
            rtval[:,:,-1,...]=field[:,:,-1,...]
    return rtval

# ========== Texture Lookup Conventions ==========



def sample_indices(field,indices):
    rtval=np.empty(indices.shape[:3]+(math.prod(field.shape[3:]),))
    field1=field.reshape(field.shape[:3]+(-1,))
    for i in range(rtval.shape[3]):
        rtval[...,i]=map_coordinates(field1[...,i],indices.transpose(3,0,1,2),order=1,mode='grid-wrap')
    rtval=rtval.reshape(indices.shape[:3]+field.shape[3:])
    return rtval


def get_positions():
    #  tex id   |  0  |  1  |  2  |  3  |  4  |
    #  uv       0     1     2     3     4     5
    x=np.linspace(.5,gridRes[0]-.5,gridRes[0])*cellSizes[0]
    y=np.linspace(.5,gridRes[1]-.5,gridRes[1])*cellSizes[1]
    z=np.linspace(.5,gridRes[2]-.5,gridRes[2])*cellSizes[2]
    xyz=np.meshgrid(x,y,z,indexing='ij')
    return np.stack(xyz,axis=-1)

def positions_to_indices(positions):
    #  tex id   |  0  |  1  |  2  |  3  |  4  |
    #  uv       0     1     2     3     4     5
    return positions/cellSizes-.5

def get_normalized_positions():
    return get_positions()/(np.asarray(gridRes)*cellSizes)
    


# ========== Display ==========
from matplotlib import pyplot as plt

def show_quiver(xyz,u,numbers=[10,10,10],**args):
    zoom=[numbers[0]/xyz.shape[0],numbers[1]/xyz.shape[1],numbers[2]/xyz.shape[2],1]
    xyz=scipy.ndimage.zoom(xyz,zoom=zoom,order=1)
    u=scipy.ndimage.zoom(u,zoom=zoom,order=1)
    plt.gca().set_aspect('equal')
    return plt.gca().quiver(xyz[...,0],xyz[...,1],xyz[...,2],u[...,0],u[...,1],u[...,2],**args)

def show_image(T,mask=None,**args):
    if mask is not None:
        args['alpha']=np.where(mask,0.,1.).swapaxes(0,1)*args.get('alpha',1)
        if not isinstance(T,np.ndarray):
            T=np.ones_like(mask)*T
    shape=T.shape[:2]
    extent=[0,shape[0]*cellSizes[0],0,shape[1]*cellSizes[1]]
    return plt.imshow(T.copy().swapaxes(0,1),extent=extent,origin='lower',**args) 


def show_terrain(xyz,elevation,**args):
    xy=xyz[:,:,0,:2]
    return plt.gca().plot_surface(xy[...,0],xy[...,1], elevation,**args)

def show_isosurface(value,rel_level=None,abs_level=None,step_res=16,**args):
    from skimage.measure import marching_cubes
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    step_size=min(gridRes)/step_res
    if len(value.shape)==4:
        value=norm(value,axis=-1)
    vmin=np.min(value.flatten())
    vmax=np.max(value.flatten())
    rel_level=rel_level or .9
    abs_level=abs_level or np.percentile(value.flatten(),rel_level*100)
    #print(vmin,vmax,level)
    if vmin<abs_level and abs_level<vmax:
        try:
            verts, faces, _, _ = marching_cubes(value.copy(), level=abs_level,spacing=cellSizes,step_size=step_size,allow_degenerate=False)
            args['color']=args.get('color','white')
            args['lw']=args.get('lw',1)
            plt.gca().plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], **args)
        except RuntimeError:
            pass
    plt.gca().set_xlim(0,cellSizes[0]*gridRes[0])
    plt.gca().set_ylim(0,cellSizes[1]*gridRes[1])
    plt.gca().set_zlim(0,cellSizes[2]*gridRes[2])
    plt.gca().set_box_aspect((cellSizes[0]*gridRes[0], cellSizes[1]*gridRes[1], cellSizes[2]*gridRes[2]))





def show_quiver_section(xy,u,numbers=[10,10],axis=0,position=None,**args):
    position=position or gridRes[axis]*cellSizes[axis]/2
    indice=np.clip(int(position/cellSizes[axis]),0,gridRes[axis]-1)
    xy=np.delete(xy.take(indices=indice,axis=axis),obj=axis,axis=-1)
    u=np.delete(u.take(indices=indice,axis=axis),obj=axis,axis=-1)
    zoom=[numbers[0]/xy.shape[0],numbers[1]/xy.shape[1],1]
    xy=scipy.ndimage.zoom(xy,zoom=zoom,order=1)
    u=scipy.ndimage.zoom(u,zoom=zoom,order=1)
    plt.gca().set_aspect('equal')
    return plt.quiver(xy[...,0],xy[...,1],u[...,0],u[...,1],**args)

def show_image_section(values,mask=None,axis=0,position=None,is_rgb=False,**args):
    if len(values.shape)==4 and not is_rgb:
        values=norm(values,axis=-1)
    position=position or gridRes[axis]*cellSizes[axis]/2
    indice=np.clip(int(position/cellSizes[axis]),0,gridRes[axis]-1)
    values=values.take(indices=indice,axis=axis)
    if mask is not None: mask=mask.take(indices=indice,axis=axis)
    cellSizes1=np.delete(cellSizes,obj=axis,axis=-1)
    if mask is not None:
        args['alpha']=np.where(mask,0.,1.).swapaxes(0,1)*args.get('alpha',1)
        if not isinstance(values,np.ndarray):
            values=np.ones_like(mask)*values
    shape=values.shape[:2]
    extent=[0,shape[0]*cellSizes1[0],0,shape[1]*cellSizes1[1]]
    plt.gca().set_aspect('equal')
    return plt.imshow(values.copy().swapaxes(0,1),extent=extent,origin='lower',**args) 


def show_contour_section(xy,values,axis=0,position=None,fmt=None,**args):
    if len(values.shape)==4:
        values=norm(values,axis=-1)
    position=position or gridRes[axis]*cellSizes[axis]/2
    indice=np.clip(int(position/cellSizes[axis]),0,gridRes[axis]-1)
    xy=np.delete(xy.take(indices=indice,axis=axis),obj=axis,axis=-1)
    values=values.take(indices=indice,axis=axis)
    plt.contourf(xy[...,0],xy[...,1],values,cmap='viridis',**args)
    contour=plt.contour(xy[...,0],xy[...,1],values,colors='black',linestyles='dashed',linewidths=1,**args)
    plt.gca().clabel(contour, contour.levels,fmt=fmt)
    plt.gca().set_aspect('equal')
    return contour