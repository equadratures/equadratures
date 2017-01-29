#!/usr/bin/env python
import Ylm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Demo 1: 
#create a triangular type spherical quadrature and plot the points
def Pranay1():
    w,theta,phi = Ylm.gen_tri_quad(10)
    x,y,z = Ylm.sph_to_cart(theta,phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,s=20*w)
    plt.show()


#Demo 2:
#same as demo 1, but with rectangular spherical quadrature
def Pranay2():
    w,theta,phi = Ylm.gen_squ_quad(10)
    x,y,z = Ylm.sph_to_cart(theta,phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,s=20*w)
    plt.show()
    
#Demo 3: check SH integration accuracy
def Pranay3():
    w,theta,phi = Ylm.gen_tri_quad(16)
    x,y,z = Ylm.sph_to_cart(theta,phi)
    err1, err2 = Ylm.test_quads(x,y,z,w,5)
    plt.plot(err1)
    plt.plot(err2)

    
Pranay2()