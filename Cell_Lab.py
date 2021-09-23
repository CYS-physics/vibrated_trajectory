# asymmetric interacting particles under active noise
# 2D periodic boundary condition
# Yunsik Choe, Seoul national University

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
import sys

import imageio
from PIL import Image

class Cell_Lab:     # OOP
    """basic model to simulate 2D passive objects under active noise"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,L, N_passive,N_active,Fs):
        
        
        # set up coefficients
        self.set_coeff(L,N_passive,N_active,Fs) 
      
        # initializing configuration of state
        self.set_zero()
        
        self.record = False
        
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,L,N_passive,N_active,Fs):
        
        # system coefficients
        self.L=L
        self.N_passive = N_passive
        self.N_active = N_active
#         self.N_ensemble = N_ensemble
        self.Fs=Fs
        self.dt = 1/Fs
        
        self.N_skip = 50
        
        # noise coefficients
        self.D = 20
        self.Dr = 20
#         self.tau_noise = 1
        
        # dynamics
        self.p = 0
        
        # inner structure coefficients
        self.AR = 1.25
        self.r = 1
        self.k = 5
        self.mu = 1
        self.mur = 0.2
        self.l = self.r*(self.AR-1)
       
        
  
    # boundary condition
    
    def periodic(self,x,y):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        mod_y = -self.L/2   +    (y+self.L/2)%self.L               # returns -L/2 ~ L/2

        return (mod_x,mod_y)
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        
        self.X_p = np.random.uniform(-self.L/2,self.L/2,self.N_passive)
        self.Y_p = np.random.uniform(-self.L/2,self.L/2,self.N_passive)
        
        self.X_a = np.random.uniform(-self.L/2,self.L/2,self.N_active)
        self.Y_a = np.random.uniform(-self.L/2,self.L/2,self.N_active)
        self.O_a = np.random.uniform(-np.pi,np.pi,self.N_active)
        
        self.set_structure()
        
       
        
    def set_structure(self):
        self.Xs = self.X_a.reshape(1,-1) + np.array([self.l,-self.l]).reshape(-1,1)*np.cos(self.O_a)
        self.Ys = self.Y_a.reshape(1,-1) + np.array([self.l,-self.l]).reshape(-1,1)*np.sin(self.O_a)
        
        (self.Xs,self.Ys) = self.periodic(self.Xs,self.Ys)
    
    
    def WCA(self,rx,ry,r_cut,k):
        r_0 = r_cut*2**(-1/6)
        r = np.sqrt(rx**2 + ry**2)
        force = 4*k*(-12*r**(-13)/r_0**(-12)+6*r**(-7)/r_0**(-6))*(np.abs(r)<r_cut)
        force = np.nan_to_num(force,copy=False)
        return force*(np.divide(rx,r,out=np.zeros_like(rx),where=r!=0),np.divide(ry,r,out=np.zeros_like(ry),where=r!=0))


    def force(self):
        fx_p = np.zeros(self.N_passive)
        fy_p = np.zeros(self.N_passive)
        fx_a = np.zeros(self.N_active)
        fy_a = np.zeros(self.N_active)
        torque = np.zeros(self.N_active)
        
        # passive passive
        relX = (self.X_p.reshape(-1,1)-self.X_p.reshape(1,-1))
        relY = (self.Y_p.reshape(-1,1)-self.Y_p.reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_p-=np.sum(fx,axis=1)
        fy_p-=np.sum(fy,axis=1)
        fx_p+=np.sum(fx,axis=0)
        fy_p+=np.sum(fy,axis=0)
        
        # active passive
        relX = (self.X_p.reshape(-1,1)-self.X_a.reshape(1,-1))
        relY = (self.Y_p.reshape(-1,1)-self.Y_a.reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_p-=np.sum(fx,axis=1)
        fy_p-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        
        relX = (self.X_p.reshape(-1,1)-self.Xs[0,:].reshape(1,-1))
        relY = (self.Y_p.reshape(-1,1)-self.Ys[0,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_p-=np.sum(fx,axis=1)
        fy_p-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        relX = (self.X_p.reshape(-1,1)-self.Xs[1,:].reshape(1,-1))
        relY = (self.Y_p.reshape(-1,1)-self.Ys[1,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_p-=np.sum(fx,axis=1)
        fy_p-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(-self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        
        # active active
        relX = (self.X_a.reshape(-1,1)-self.X_a.reshape(1,-1))
        relY = (self.Y_a.reshape(-1,1)-self.Y_a.reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        
        relX = (self.X_a.reshape(-1,1)-self.Xs[0,:].reshape(1,-1))
        relY = (self.Y_a.reshape(-1,1)-self.Ys[0,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        relX = (self.X_a.reshape(-1,1)-self.Xs[1,:].reshape(1,-1))
        relY = (self.Y_a.reshape(-1,1)-self.Ys[1,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(-self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        relX = (self.Xs[0,:].reshape(-1,1)-self.Xs[0,:].reshape(1,-1))
        relY = (self.Ys[0,:].reshape(-1,1)-self.Ys[0,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        torque-=np.sum(self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        relX = (self.Xs[0,:].reshape(-1,1)-self.Xs[1,:].reshape(1,-1))
        relY = (self.Ys[0,:].reshape(-1,1)-self.Ys[1,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        torque-=np.sum(self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(-self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        relX = (self.Xs[1,:].reshape(-1,1)-self.Xs[1,:].reshape(1,-1))
        relY = (self.Ys[1,:].reshape(-1,1)-self.Ys[1,:].reshape(1,-1))
        
        (relX,relY) = self.periodic(relX,relY)
        (fx,fy) = self.WCA(relX,relY,2*self.r,self.k)
        fx_a-=np.sum(fx,axis=1)
        fy_a-=np.sum(fy,axis=1)
        torque-=np.sum(-self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=1)
        fx_a+=np.sum(fx,axis=0)
        fy_a+=np.sum(fy,axis=0)
        torque+=np.sum(-self.l*(fy*np.cos(self.O_a)-fx*np.sin(self.O_a)),axis=0)
        
        
        
        
        
        return(fx_p,fy_p,fx_a,fy_a,torque)

    
    def time_evolve(self):
        
        # compute force & torque
        (fx_p,fy_p,fx_a,fy_a,torque) = self.force()
        
        # noise
        fx_p+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_passive)
        fy_p+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_passive)
#         fx_a+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_active)
#         fy_a+=np.random.normal(0,np.sqrt(2*self.D/self.dt),self.N_active)
#         torque+=np.random.normal(0,np.sqrt(2*self.Dr/self.dt),self.N_active)

        # update configuration


        self.X_p += self.mu*fx_p*self.dt
        self.Y_p += self.mu*fy_p*self.dt
        self.X_a += self.mu*(fx_a+self.p*np.cos(self.O_a))*self.dt
        self.Y_a += self.mu*(fy_a+self.p*np.sin(self.O_a))*self.dt
        self.O_a += self.mur*torque*self.dt

        (self.X_p,self.Y_p) = self.periodic(self.X_p,self.Y_p)
        (self.X_a,self.Y_a) = self.periodic(self.X_a,self.Y_a)
        self.set_structure()
        
    def initialize(self):
        self.set_zero()
        L = self.L
        self.L = 2*L
        for i in range(5000):
            self.L = (2-(i/5000))*L
            self.time_evolve()
        self.L = L
        
        for _ in range(10000):
            self.time_evolve()
        
    def animate(self,N_iter,directory):
        self.initialize()
        
        axrange = [-self.L/2, self.L/2, -self.L/2, self.L/2]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(self.X_p,self.Y_p,color='blue')
        ax1.scatter(self.X_a,self.Y_a,color='red')
        ax1.scatter(self.Xs,self.Ys,color='red')

        ax1.axis(axrange)
        ax1.set_aspect('equal', 'box')
        fig1.show()
        fig1.canvas.draw()
        
        os.makedirs('record/'+str(directory),exist_ok=True)


        
        
        for nn in trange(N_iter):
            
            ax1.clear()
            ax1.scatter(self.X_p,self.Y_p,s=self.r**2*200000/self.L**2,color='blue')
            ax1.scatter(self.X_a,self.Y_a,s=self.r**2*200000/self.L**2,color='red')
            ax1.scatter(self.Xs,self.Ys,s=self.r**2*200000/self.L**2,color='red')

            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')
            fig1.show()
            fig1.canvas.draw()
            fig1.savefig(str(os.getcwd())+'/record/'+str(directory)+'/'+str(nn)+'.png')
            for _ in range(self.N_skip):
                self.time_evolve()
                
    
            
                
    def measure(self,N_iter):
        self.initialize()
        x_traj = np.zeros(N_iter)
        y_traj = np.zeros(N_iter)
        o_traj = np.zeros(N_iter)
        for nn in range(N_iter):
            self.time_evolve()
            x_traj[nn] = self.X_a
            y_traj[nn] = self.Y_a
            o_traj[nn] = self.O_a
        axrange = [-self.L/2, self.L/2, -self.L/2, self.L/2]
        plt.scatter(x_traj,y_traj,s=1)
        plt.axis(axrange)
        plt.show()
        return(x_traj,y_traj,o_traj)
    
