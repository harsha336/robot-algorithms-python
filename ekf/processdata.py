#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:44:06 2019

@author: sreeharshacs

Processing functions for the dataset in the encapsulating folder.
Part of Prof. S Carpin's lectures

inputs.txt - contains continuous actions with 0.5 seconds deltaT
sporadic_sensor_readings.txt - Non continous sensor readings
sensor_readings.txt - Continuous sensor readings
"""

from ekfpy import ekf
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import ploterrellips as pe

"""
Sporadic readings - Non continous sensor readings
"""
def processSporadicReadings():
    # Read data files and init a data frame
    action = pd.read_csv('../data/inputs.txt',header=None,sep=" ",dtype=float)
    action = action.dropna(axis=1,how='all')
    sensor_reading = pd.read_csv('../data/sporadic_sensor_readings.txt',header=None,sep=" ",dtype=float)
    sensor_reading = sensor_reading.dropna(axis=1,how='all')
    
    # init time
    inp_time = 0.5
    
    # initialize the ekf object. The pose is assumed to be 0,0,0
    e = ekf()
    pose = [[0.0,0.0,0.0]]
    
    # initialize the sensor readings iterator
    zit = iter(sensor_reading.iterrows())
    it,z = next(zit)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for temp in action.itertuples():
        te = temp[1:4]
        u = [t for t in te if not math.isnan(t)]
        
        # There is a sensor reading for this time, so call update step along
        # with predict step
        if inp_time == z[z.index[0]]:
            temp = [t for t in z]
            e.performEkfStep(0.5,u,1,z[1:])
            pose.append([e.mu[0],e.mu[1],e.mu[2]])
            try:
                it,z = next(zit)
            except StopIteration:
                print("No more sensor readings!")  
        
        # No sensor reading, calling only predict step
        else:
            e.performEkfStep(0.5,u,0)
            pose.append([e.mu[0],e.mu[1],e.mu[1]])
        
        # Plot the error ellipsoid
        mean = [e.mu[0],e.mu[1]]
        sigma = e.sigma[:-1,:-1]
        pe.plot_cov_ellipse(sigma, mean, nstd=2,ax=ax)
        inp_time += 0.5
    
    # Plot the path
    posearr = np.array(pose)
    plt.plot(posearr[:,0],posearr[:,1])
  
"""
Continuous readings
"""
def processContinuousReadings():
    # Read data files and init a data frame
    action = pd.read_csv('../data/inputs.txt',header=None,sep=" ",dtype=float)
    action = action.dropna(axis=1,how='all')
    sensor_reading = pd.read_csv('../data/sensor_readings.txt',header=None,sep=" ",dtype=float)
    sensor_reading = sensor_reading.dropna(axis=1,how='all')
    
    # Concatenating two dfs as there is a sensor reading at each time step
    asr = pd.concat([action,sensor_reading],axis=1)
    
    # initialize the ekf object. The pose is assumed to be 0,0,0
    e = ekf([0,0,0])
    pose = [[0,0,0]]
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    # Iterate all inputs and sensor readings.
    for uz in asr.itertuples(): 
        
        # Filtering out the NaN values. The df seems to be improperly constructed.
        temp = uz[1:4]
        u = [t for t in temp if not math.isnan(t)]
        temp = uz[4:]
        z = [t for t in temp if not math.isnan(t)]
        
        # Perform predict and update step always
        e.performEkfStep(0.5,u,1,z)
        pose.append([e.mu[0],e.mu[1],e.mu[2]])
        
        # Plot the error ellipsoid
        mean = [e.mu[0],e.mu[1]]
        sigma = e.sigma[:-1,:-1]
        pe.plot_cov_ellipse(sigma, mean, nstd=2,ax=ax)
    
    # Plot the path
    posearr = np.array(pose)
    plt.plot(posearr[:,0],posearr[:,1])