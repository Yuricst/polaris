#!/usr/bin/env python3
"""Function for separating coordinates"""

import numpy as np

def plotCircle(centerX, centerY, radius, steps=300):
    """Function to plot circle with user defined center coordinate and radius
    
    Args:
        centerX (float): origin x-coordinate
        centerY (float): origin y-coordinate
        radius (float): radius of circle
    Returns:
        (tuple): tuple of x and y coordinates of the circle
    """
    # initialize
    x_arr = np.zeros((steps,))
    y_arr = np.zeros((steps,))
    theta = np.linspace(0, 2*np.pi, steps)
    for i in range(len(x_arr)):
        x_arr[i] = radius * np.cos(theta[i]) + centerX
        y_arr[i] = radius * np.sin(theta[i]) + centerY
    return x_arr, y_arr
    

def plotSphere(centerX, centerY, centerZ, radius):
    """Function to plot sphere using ax.plot_wireframe(x,y,z)
    
    Args:
        centerX (float): origin x-coordinate
        centerY (float): origin y-coordinate
        centerZ (float): origin z-coordinate
        radius (float): radius of circle
    Returns:
        (tuple): tuple of x, y, and z coordinates of a sphere
    """
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u)*np.sin(v) + centerX
    y = radius*np.sin(u)*np.sin(v) + centerY
    z = radius*np.cos(v) + centerZ
    
    return x, y, z