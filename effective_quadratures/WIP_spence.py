import numpy as np
import scipy.special as sp
import copy as cp

def fn_orpolyval(x,n,an,bn,orpoly):
    if n == 0:
        p1 = (x-an[0])*orpoly[:,0]
    else:
        p1 = (x-an[n])*orpoly[:,n]-bn[n]*orpoly[:,n-1]
    return p1
    
def fn_erfun_quad(a,b,h,tol):
    abscissa = lambda k,h,a,b: 0.5*((b-a)*sp.erf(k*h)+(a+b))
    weight = lambda k,h,a,b: (b-a)*np.exp(-(k*h)**2)/np.sqrt(np.pi)
    k = 0
    uk1 = np.array([abscissa(k,h,a,b)])
    wgt = weight(k,h,a,b)
    wk1 = np.array([wgt])
    while wgt > tol:
        k = k+1
        uk1 = np.append(uk1,[abscissa(k,h,a,b),abscissa(-k,h,a,b)])
        wgt = weight(k,h,a,b)
        wk1 = np.append(wk1,[wgt,wgt])
    
    lgt = 2*k+1
    m = np.arange(1,k+1,1)    
    uk = np.zeros(lgt)
    wk = np.zeros(lgt)
    cz = k+1
    uk[cz-1] = uk1[0]
    wk[cz-1] = wk1[0]
    ixp = cz+m
    ixm = cz-m
    ixp1 = 2*m
    ixm1 = ixp1+1
    uk[ixp-1] = uk1[ixp1-1]
    uk[ixm-1] = uk1[ixm1-1]
    wk[ixp-1] = wk1[ixp1-1]
    wk[ixm-1] = wk1[ixm1-1]
    return uk, wk
    
def fn_qrs_public(n,flag):
    h = 0.004
    tol = 1e-17
    if flag==1: #QRS45_azimuthal, [0,pi/2], pi/4 symmetry
        a = -1./np.sqrt(2)
        b = -a
        wfn = lambda u: 1./np.sqrt(1-u**2)
        uk, wk = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==2: #QRS60_azimuthal, [0,pi/3], pi/6 symmetry
        a = -1./2
        b = -a
        wfn = lambda u: 1./np.sqrt(1-u**2)
        uk, wk = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==3: #QRS90_azimuthal, [0,pi/2], pi/2 symmetry
        a = 0
        b = np.pi/2
        wfn = lambda u: 1
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = 0.5*np.sin(uk)
    elif flag==4: #QRA90_azimuthal_2D_RZ
        a = 0
        b = 1
        wfn = lambda u: 2./np.sqrt(2-u**2)
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk*np.sqrt(2-uk**2)
    elif flag==5: #QRA45_azimuthal, [0,pi/2], pi/4 symmetry (sin(0.5*(beta-pi/4)))
        a = -np.sqrt(1-1./np.sqrt(2))/np.sqrt(2)
        b = -a
        wfn = lambda u: 1./np.sqrt(1-u**2)
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==6: #QRA30_azimuthal,[0,pi/3], pi/6 symmetry (sin(0.5*(beta-pi/6)))
        a = -np.sqrt(1-np.sqrt(3)/2)/np.sqrt(2)
        b = -a
        wfn = lambda u: 1./np.sqrt(1-u**2)
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==7: #Gauss-Legendre [0,pi/2], pi/4 symmetry
        a=-1
        b = 1
        wfn = lambda u: 1
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==8: #QRS90_polar 2DRZ/XYZ, [0,pi/2],pi/2 symmetry (cos(alpha))
        a = 0
        b = 1
        wfn = lambda u: 1
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk        
    elif flag==9: #QRS90b_polar 2DRZ/XYZ, [0,pi/2],pi/2 symmetry (sin(alpha))
        a = 0
        b = 1
        wfn = lambda u: 1
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = np.sqrt(1-uk**2)
    elif flag==10: #QRS45_polar, [0,pi/2], pi/4 symmetry
        a = -1./np.sqrt(2)
        b = -a
        wfn = lambda u: (u+np.sqrt(1-u**2))/(np.sqrt(2)*np.sqrt(1-u**2))
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==11: #QRA45_polar, [0,pi/2], pi/4 symmetry
        a = -np.sqrt(1-1./np.sqrt(2))/np.sqrt(2)
        b = -a
        wfn = lambda u: np.sqrt(2)*((1-2*u**2)+2*u*np.sqrt(1-u**2))/np.sqrt(1-u**2)
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = uk
    elif flag==12: #QRA90_2D_polarXY
        a = 0
        b = 1
        wfn = lambda u: 1
        [uk,wk] = fn_erfun_quad(a,b,h,tol)
        tuk = np.sqrt(1-uk**2)

    #get the recurrence relation coefficients        
    l = len(uk)
    iprod = np.zeros(n)
    iprodx = np.zeros(n)
    an = np.zeros(n)
    bn = np.zeros(n)
    orpoly = np.zeros((l,n))
    orpoly[:,0] = 1 #set P0(x) = 1
    integx = wfn(uk)*tuk*orpoly[:,0]**2
    integ = wfn(uk)*orpoly[:,0]**2
    iprodx[0] = h*sum(wk*integx)
    iprod[0] = h*sum(wk*integ)
    an[0] = iprodx[0]/iprod[0]
    bn[0] = 0
    for i in range(n-1):
        orpoly[:,i+1] = fn_orpolyval(tuk,i,an,bn,orpoly)
        integx = wfn(uk)*tuk*orpoly[:,i+1]**2
        integ = wfn(uk)*orpoly[:,i+1]**2
        iprodx[i+1] = h*sum(wk*integx)
        iprod[i+1] = h*sum(wk*integ)
        an[i+1] = iprodx[i+1]/iprod[i+1]
        bn[i+1] = iprod[i+1]/iprod[i]
    
    #Golub-Welsch
    vd = cp.deepcopy(an)
    vod = np.sqrt(bn[1:n])
    jmat = np.diag(vod,-1)+np.diag(vd) + np.diag(vod,1)
    vals, vecs = np.linalg.eigh(jmat)
    wi = iprod[0]*vecs[0,:]**2
    
    #return abscissas
    if flag == 1:
        xi = np.arcsin(vals)+np.pi/4
    elif flag == 2:
        xi = np.arcsin(vals)+np.pi/6
    elif flag == 3:
        xi = np.arcsin(2*vals)
    elif flag == 4:
        xi = np.arccos(vals)
        xi = np.flipud(xi)
        wi = np.flipud(wi)
    elif flag == 5:
        xi = 2*(np.arcsin(vals)+np.pi/8)
        wi = 2*wi
    elif flag == 6:
        xi = 2*(np.arcsin(vals)+np.pi/12)
        wi = 2*wi
    elif flag == 7:
        xi = np.pi/4*(vals+1)
    elif flag == 8:
        idx = np.argsort(np.arccos(vals))
        xi = np.arccos(vals[idx])
        wi = wi[idx]
    elif flag == 9:
        xi = np.arcsin(vals)
    elif flag == 10:
        xi = np.arcsin(vals)+np.pi/4
    elif flag == 11:
        xi = 2*np.arcsin(vals)+np.pi/4
    elif flag == 12:
        xi = np.arcsin(vals)
        
    return wi,xi
    
