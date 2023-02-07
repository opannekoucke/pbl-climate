'''

    ATTENTION : ce script ne doit pas être modifié durant le TP, il n'est
                pas demandé de comprendre l'organisation du script.

Description:
------------
    package pour le TP Lorenz 1a

Auteur:
-------
    O. Pannekoucke (INPT-ENM, CNRM, CERFACS)

Historique:
-----------
    4/10/2017   :   création
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Model(object):
    ''' Template of dynamical model '''
    def __init__(self):
        raise NotImplementedError()
    
    def _trend(self,t,x):
        raise NotImplementedError()
    
    def _time_scheme(self,t,x):
        raise NotImplementedError()
        
    def _euler(self,tq, x0):
        return x0 + self.dt * self._trend(tq,x0)
    
    def _rk4(self,tq,x0):
        ''' Fourth order Runge-Kutta time integration'''
        dt = self.dt
        k1 = dt * self._trend( tq, x0 )
        k2 = dt * self._trend( tq+0.5*dt, x0+k1*0.5 )
        k3 = dt * self._trend( tq+0.5*dt, x0+k2*0.5 )
        k4 = dt * self._trend( tq+dt, x0+k3 )
        return x0 + (k1 + k4)/6 + (k2+k3)/3
        
    def forecast(self,t,x0):
        ''' Time integration loop '''        
        traj=[x0]
        for tq in t[:-1]:
            x1 = self._time_scheme( tq, x0 )
            traj.append(x1)
            x0 = x1            
        return traj    

class Lorenz(Model):
    
    def __init__(self, r = 32, sigma = 10.0, b = 8/3, f = 0., dt = 0.01):
        # set parameters for Lorenz 1963
        self.r = r
        self.sigma = sigma
        self.b = b
        self.f = f
        # Set time step for integration
        self.dt = dt
        # Set the strategy for time scheme
        self._time_scheme = self._rk4
    
    def _trend(self,t,x):
        ''' Compute the rend of the dynamical system '''
        X,Y,Z = x
        dX = self.sigma*(Y-X) + self.f
        dY = (self.r-Z)*X-Y   + self.f
        dZ = X*Y-self.b*Z
        return np.array([dX, dY, dZ])
    
    def forecast(self,t,x0):
        traj = super().forecast(t,x0)
        return LorenzTrajectory(t,np.array(traj))

class LorenzTrajectory(object):
    
    def __init__(self,t,data):
        self.t=t
        self.t_anim=t[::10]
        self.data=data
        self.data_anim=data[::10,:]

    def __call__(self,i):
        return self.data[i,:]

    def __add__(self,traj2):
        if len(self.t)==len(traj2.t):
            return LorenzTrajectory(self.t,self.data + traj2.data)
        else:
            raise Exception("Trajectory Addition Error : not same time length")

    def __sub__(self,traj2):
        if len(self.t)==len(traj2.t):
            return LorenzTrajectory(self.t,self.data - traj2.data)
        else:
            raise Exception("Trajectory Addition Error : not same time length")    
            
    def plot_X(self):
        #plt.figure()
        plt.plot(self.t, [state[0] for state in self.data] )
        ax = plt.gca()
        ax.set_title('$X(t)$')
        ax.set_xlabel('t')
        ax.set_ylabel('Magnitude of $X(t)$')
        #return
    
    def plot_norm(self):
        plt.figure()
        norm = np.sqrt( (self.data**2).sum(1) )
        plt.semilogy(self.t, norm)
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$|| x(t)||$")
        plt.title("Norm (semi-log)")
    
    def plot_time_series(self):
        #plt.figure()
        labels=["$X(t)$","$Y(t)$","$Z(t)$"]
        for i,label in zip(range(3),labels):
            plt.plot(self.t, self.data[:,i],label=label )
        ax = plt.gca()
        ax.set_xlabel('t')
        ax.set_ylabel(f'Magnitude of modes')
        plt.legend()
        return
    
    def plot_histogram(self):
        labels=["$X(t)$","$Y(t)$","$Z(t)$"]
        num_bins=100
        for i,label in zip(range(3),labels):
            fig,ax=plt.subplots()
            n, bins, patches=ax.hist(self.data[:,i],num_bins)
            ax = plt.gca()
            ax.set_xlabel(f"{label}")
            ax.set_ylabel(f"Frequency {label}")
            plt.title(f"Histogram for {label}")
        return
    
    def plot_3d(self,figsize=(12,12)):
        from mpl_toolkits.mplot3d import Axes3D
        #fig = plt.figure(figsize=figsize)
        #ax = fig.gca(projection='3d')
        ax = plt.gca(projection='3d')
        traj=np.array(self.data)
        xs,ys,zs  = traj[:,0], traj[:,1], traj[:,2]
        ax.plot(xs, ys, zs, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.show()
        return

    def anim_old(self):
        # Variable statique
        # parameters for grid
        h=1.0
        npt=20
        a = 1.
        x = np.linspace(-2,2,2*npt)
        z = np.linspace(0,1,npt)
        [x,z] = np.meshgrid(x,z,indexing='ij')
        
        # Build spatial modes
        Vx = -np.sin(np.pi/h*x)*np.cos(np.pi/h*z) 
        Vz =  np.cos(np.pi/h*x)*np.sin(np.pi/h*z)
        ThY = np.sqrt(2)*np.cos(np.pi*a*x/h)*np.sin(np.pi*z/h)
        ThZ = np.sin(2*np.pi*z/h)
        
        fig=plt.figure()
        fig.show()
        for state,t in zip(self.data,self.t):
            X,Y,Z = state
            # Velocity and temperature
            vx, vz = X*Vx, X*Vz
            th = Y*ThY - Z*ThZ
            # set normalization
            v_max=2.
            vx[0,0] = - v_max
            vz[0,0] = 0.
            th[0,0] = 1.
            # plot
            plt.clf()
            plt.contourf(x,z,th)
            plt.quiver(x,z,vx,vz)
            plt.title(f"time: {t}")
            plt.draw()
            plt.pause(0.01)
        return
    
    def video(self, filename='./img/lorenz.mp4'):
        import io
        import base64
        from IPython.display import HTML
        video = io.open(filename, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))        
    
    def build_video(self, filename="./img/lorenz.mp4"):
        lanim=self.animL80()
        print("Video: under construction")
        lanim.save(filename)
        print("Video: created")
        return

    @property
    def psi(self):
        npt = 50
        x = np.linspace(-8000,8000,2*npt)
        y = np.linspace(-4000,4000,npt)
        [x,y] = np.meshgrid(x,y,indexing='ij')
        self._x = x
        self._y = y

        # Build spatial modes
        L = 1080
        k1=np.array([0, 1])/L
        k2=np.array([np.cos(np.pi/6), np.sin(np.pi/6)])/L
        k3=-k1-k2;

        Psi1=np.cos(k1[0]*x+k1[1]*y);
        Psi2=np.cos(k2[0]*x+k2[1]*y);
        Psi3=np.cos(k3[0]*x+k3[1]*y);
        Psi = np.array([Psi1, Psi2, Psi3])    
        self._psi = Psi
        return self._psi
    
    def animL80(self):
        '''
        Animation of solution for L63 version L80
        '''        
        import matplotlib.animation as animation
        
        def update_plot(i,traj,graphe):
            '''
            Local iteration used within animate package.
            '''
            X,Y,Z = traj.data_anim[i]
            t = traj.t_anim[i]
            # Geopotential
            psi1=1/(A*B*h1)*Z+F1/(nu0*(g0+1));
            psi2=1/(A*B*h1)*Y;
            psi3=1/A*X;
            psi = psi1*graphe.Psi[0] + psi2*graphe.Psi[1] + psi3*graphe.Psi[2]
                        
            graphe.ax.clear()
            plt.contour(graphe.x,graphe.y, psi, graphe.zlevels)
            plt.xlabel('x')
            plt.ylabel('y')            
            plt.title(f'time: {t:.2f}')
            
            return
        
        class Graph(object): pass
       
        h=1.0
        npt=20
        a = 1.

        g0=8.0
        k0=1.0/48.0
        nu0=k0
        a = np.array([1, 1, 3.0])
        #h = [-1 0 0]';  
        h1=-1
        #f = [0.1 0 0]'; 
        F1=0.108

        b=np.zeros(3);
        b[0] = 0.5*(a[0]-a[1]-a[2]);
        b[1] = 0.5*(a[1]-a[2]-a[0]);
        b[2] = 0.5*(a[2]-a[0]-a[1]);
        c = np.sqrt( b[0]*b[1] + b[1]*b[2] + b[2]*b[0] );

        F1c=0.10785
        a3=3
        L=1080


        A=1/(nu0*(g0+1))*g0*c*(a3-1);
        B=1/(nu0*(a3*g0+1))*c/a3;        

        print("Animation: under construction")
        graphe=Graph()
        
        # Variable statique
        # parameters for grid
        #    psi1=1/(A*B*h1)*Z+F1/(nu0*(g0+1));
        #    psi2=1/(A*B*h1)*Y;
        #    psi3=1/A*X;       
        graphe.zlevels = np.linspace(-2,2,20)*(1/A + 1/(A*B*h1)*2+F1/(nu0*(g0+1)))
    
        graphe.Psi = self.psi
        graphe.x = self._x
        graphe.y = self._y
        
        graphe.fig, graphe.ax=plt.subplots()
        graphe.geopotentiel = plt.contour(graphe.x,graphe.y, graphe.Psi[0])


        plt.xlabel('x')
        plt.ylabel('y')
        nstep=len(self.t_anim)        
        lanim=animation.FuncAnimation(graphe.fig, update_plot,
                    frames=nstep, fargs=(self,graphe),
                                 interval=100, blit=False,repeat=False)
            
        #plt.show(graphe.fig)
        print("Animation: created")
        #return graphe.fig
        return lanim
   

######################################################
#   
######################################################


def make_simulation():
    # Build instance of Lorenz model (default parameter are used)
    model = Lorenz()

    # Set initial state
    x0_long = np.array([1.0, 0.0, 0.0])

    # Long time integration 
        # set time
    ndt = 3000
    t_long = model.dt * np.arange(ndt)
        # Compute trajectory
    long_traj = model.forecast(t_long,x0_long)

    # Short time integration
    ndt = 2000
    t_short = model.dt * np.arange(ndt)
    x0_short = long_traj(-1)
    short_traj = model.forecast(t_short, x0_short )
    return short_traj


def interactive_lorenz(r=32,X=1., Y=0,Z=0):
    # set time window
    dt = 0.01
    Tmax=300
    ndt = Tmax//dt
    t = np.arange(ndt)*dt
    
    # Initial state
    start = np.array([X,Y,Z])
    
    # compute trajectory
    model = Lorenz(r=r)
    traj = model.forecast(t, start)
    traj = traj.data
    
    # 3D plot simulation

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111,projection='3d')
    
    traj=np.array(traj)
    xs,ys,zs  = traj[:,0], traj[:,1], traj[:,2]
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Build instance of Lorenz model (default parameter are used)
    model = Lorenz()
        
    # Set initial state
    x0_long = np.array([1.0, 0.0, 0.0])

    # Long time integration 
    # set time
    ndt = 3000
    t_long = model.dt * np.arange(ndt)
        # Compute trajectory
    long_traj = model.forecast(t_long,x0_long)

    # Short time integration
    ndt = 2000
    t_short = model.dt * np.arange(ndt)
    x0_short = long_traj(-1)
    #short_term_traj = model.forecast(t_short, x0_short )    


    # Set initial state
    ndt = 30000
    long_term_window = model.dt * np.arange(ndt)

    # Compute trajectory
    long_term_traj = model.forecast(long_term_window, x0_short )    

    long_term_traj.plot_3d()