import numpy as np
import spence as s
import scipy.special as sp


def Ylm(l,m,mu,phi):
    if np.any(abs(m) > l) or l < 0:
        raise(ValueError,'invalid order/degree input')
    elif np.any(abs(mu) > 1):
        raise(ValueError,'invalid polar cosine input')
    
    norm = (-1)**m*np.sqrt((2-(m==0))*(2*l+1)/(4*np.pi)*sp.factorial(l-abs(m))/sp.factorial(l+abs(m)))

    P = sp.lpmv(abs(m),l,mu)
    if m < 0:
        azi = (-1)**m*np.sin(abs(m)*phi)
    else:
        azi = np.cos(m*phi)
        
    return norm*P*azi

def test_quads(xi,eta,mu,w,order):
    phi = np.arctan2(eta,xi)    
    err1 = np.zeros((order+1)**2)
    err2 = np.zeros((order+1)**2)
    
    for l in range(order):
        for m in range(2*l+1):
            harms = Ylm(l,m-l,mu,phi)
            err1[l**2+m] = sum(w*harms)
            err2[l**2+m] = sum(w*(harms*harms))-1
    err1[0] = err1[0] - np.sqrt(4*np.pi)
    return err1, err2

def gen_tri_quad(n):
    wt,theta = s.fn_qrs_public(n,10)
    THETA = np.zeros(n*(n+1)/2)
    PHI = np.zeros(n*(n+1)/2)
    W = np.zeros(n*(n+1)/2)
    count = 0
    for i in range(n):
        wp,phi = s.fn_qrs_public(i+1,5)
        for j in range(i+1):
            THETA[count] =theta[i]
            PHI[count] = phi[j]
            W[count] = wt[i]*wp[j]
            count = count+1
    theta8 = np.zeros(8*len(THETA))
    phi8 = np.zeros(8*len(PHI))
    w8 = np.zeros(8*len(W))
    for k in range(len(W)):
        theta8[8*k:8*k+4] = THETA[k]
        theta8[8*k+4:8*k+8] = np.pi-THETA[k]
        phi8[8*k:8*k+8] = PHI[k] + np.pi*np.array([0,0.5,1.,1.5,0.,0.5,1.,1.5])
        w8[8*k:8*k+8] = W[k]
        
    return w8,theta8,phi8
    
def gen_squ_quad(n):
    wt, theta = s.fn_qrs_public(n,10)
    wp, phi = s.fn_qrs_public(n,5)
    THETA = np.zeros(n**2)
    PHI = np.zeros(n**2)
    W = np.zeros(n**2)
    count = 0
    for i in range(n):
        for j in range(n):
            THETA[count] = theta[i]
            PHI[count] = phi[j]
            W[count] = wt[i]*wp[j]
            count = count + 1
    
    theta8 = np.zeros(8*len(THETA))
    phi8 = np.zeros(8*len(PHI))
    w8 = np.zeros(8*len(W))
    for k in range(len(W)):
        theta8[8*k:8*k+4] = THETA[k]
        theta8[8*k+4:8*k+8] = np.pi-THETA[k]
        phi8[8*k:8*k+8] = PHI[k] + np.pi*np.array([0,0.5,1.,1.5,0.,0.5,1.,1.5])
        w8[8*k:8*k+8] = W[k]
        
    return w8,theta8,phi8
    
def sph_to_cart(theta,phi):
    xi = np.sin(theta)*np.cos(phi)    
    eta = np.sin(theta)*np.sin(phi)
    mu = np.cos(theta)
    return xi, eta, mu