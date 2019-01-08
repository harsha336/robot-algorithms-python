#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 20:13:41 2019

@author: sreeharshacs

The Extended Kalman Filter implementation
"""

import numpy as np
from numpy import sin,cos
from numpy.linalg import multi_dot

class ekf:
    
    # Error in motion model
    R = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.001]])
    
    # Error in sensor model
    Q = np.array([[0.001,0,0],[0,0.001,0],[0,0,0.001]])
    
    # Constructor - Default initial pose is (0,0,0)
    def __init__(self,init_pose = [0.0,0.0,0.0]):
        self.mu = np.array(init_pose)
        self.sigma = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]])
    
    """
    The kalman filter step based on the delta time passed.
    If puflag is set, then there is a sensor reading and hence perform update
    step. Else perform only predict step.
    """    
    def performEkfStep(self,deltaT,u,puflag,*z):
        
        # Linearization function
        G = np.array([[1,0,u[0]*deltaT*(-sin(self.mu[2]))],
                       [0,1,u[0]*deltaT*cos(self.mu[2])],
                       [0,0,1]])
        
        # Predict Step
        mubar = np.array([(self.mu[0] + u[0]*deltaT*cos(self.mu[2])),
                          (self.mu[1] + u[0]*deltaT*sin(self.mu[2])),
                          (self.mu[2] + u[1]*deltaT)])
        mubar = mubar.transpose()
        sigmabar = np.dot(G,self.sigma)
        sigmabar = np.dot(sigmabar,G.transpose())
        
        # Add motion model error covariance
        sigmabar += ekf.R
        
        if not puflag:
            self.mu = mubar
            self.sigma = sigmabar
        
        # Update Step
        else:
            
            # Sensor model derivatives
            H = np.array([[(2*(mubar[0])-10)/(2*np.sqrt((mubar[0]-5)**2+(mubar[1]-5)**2)),
                           (2*(mubar[1])-10)/(2*np.sqrt((mubar[0]-5)**2+(mubar[1]-5)**2)),
                           0],
                          [(2*(mubar[0])-8)/(2*np.sqrt((mubar[0]-4)**2+(mubar[1]-7)**2)),
                           (2*(mubar[1])-14)/(2*np.sqrt((mubar[0]-4)**2+(mubar[1]-7)**2)),
                           0],
                          [(2*(mubar[0])+6)/(2*np.sqrt((mubar[0]+3)**2+(mubar[1]-2)**2)),
                           (2*(mubar[1])-4)/(2*np.sqrt((mubar[0]+3)**2+(mubar[1]-2)**2)),
                           0]])
            
            # Compute kalman gain
            K_nu = np.dot(sigmabar,H.transpose())
            K_de = multi_dot([H,sigmabar,H.transpose()])
            K_de += ekf.Q
            K_de = np.linalg.inv(K_de)
            K = np.dot(K_nu,K_de)
            
            # This is based on the sensor model - predict the sensor readings
            h_mu = np.array([np.sqrt((mubar[0]-5)**2+(mubar[1]-5)**2), 
                             np.sqrt((mubar[0]-4)**2+(mubar[1]-7)**2), 
                             np.sqrt((mubar[0]+3)**2+(mubar[1]-2)**2)])
            h_mu = h_mu.transpose()
            z_np = np.array(z)
            t = (z_np-h_mu).reshape(3,1)
            
            # Add the kalman gain
            self.mu = mubar + np.dot(K,t).reshape(1,3)
            self.mu = np.squeeze(np.asarray(self.mu))
            self.sigma = sigmabar - multi_dot([K,H,sigmabar])