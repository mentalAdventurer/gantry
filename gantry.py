import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from pysim import  DynamicSystem, CompoundSystem,ContinuousLinearizedSystem, InputConnector
from mimo import poly_transition


        
class Gantry(DynamicSystem):
    """Modell des Brückenkrans"""

    def __init__(self):
        super().__init__()
        self.L1=1.0
        self.L2=1.0
        self.L=np.sqrt(self.L1**2+self.L2**2)/2
        self.M=10
        self.J=self.M/12*(self.L1**2+self.L2**2)
        self.m=0.8
        self.k=100.0

        #Eingänge anlegen
        self._inputs["accell_axis"]=InputConnector(2)

        #Ausgänge registrieren
        self._outputs["accell_frame"]=self.accell_frame
        self._outputs["gyro_frame"]=self.gyro_frame
        self._outputs["position_axis"]=self.position_axis
        self._outputs["position_total"]=self.position_total
        self._outputs["state"]=lambda t,x: self.extract_system_state(x)
        

    def linearize(self,equi,debug=False):
        """Linearization of Gantry System.

        Args:
        self: the instance
        equi: the equilibrium

        Returns:
        str: The result of the addition
        """

        #Positionen auslesen
        x1=equi[0]
        x2=equi[1]

        #Ruhelagen für alle Variablen angeben
        x_equi=np.zeros((self.number_of_states(),))
        x_equi[0]=x1
        x_equi[1]=x2
        u_equi={'accell_axis':np.zeros((2,))}
        y_equi={'position_total':x_equi[0:2],
                'position_axis':x_equi[0:2],
                'gyro_frame':np.zeros((1,)),
                'accell_frame':np.zeros((2,)),
                'state':x_equi}

        C={'position_total':np.zeros((2,self.number_of_states())),
           'position_axis':np.zeros((2,self.number_of_states())),
           'accell_frame':np.zeros((2,self.number_of_states())),
           'gyro_frame':np.zeros((1,self.number_of_states())),
           'state':np.eye(self.number_of_states())}
        D={'position_total':{'accell_axis':np.zeros((2,2))},
           'position_axis':{'accell_axis':np.zeros((2,2))},
           'accell_frame':{'accell_axis':np.zeros((2,2))},
           'gyro_frame':{'accell_axis':np.zeros((1,2))},
           'state':{'accell_axis':np.zeros((self.number_of_states(),2))}}
        B={'accell_axis':np.zeros((self.number_of_states(),2))};
        
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier die sollten die korrekten Matrizen angegeben werden
        A=np.zeros((10,10))
        B['accell_axis']=np.zeros((10,2))
        C['position_total']=np.zeros((2,10))
        C['position_axis']=np.zeros((2,10))
        C['gyro_frame']=np.zeros((1,10))
        C['accel_frame']=np.zeros((2,10))
        D['position_total']['accell_axis']=np.zeros((2,2))
        D['position_axis']['accell_axis']=np.zeros((2,2))
        D['gyro_frame']['accell_axis']=np.zeros((1,2))
        D['accell_frame']['accell_axis']=np.zeros((2,2))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return ContinuousLinearizedSystem(A,B,C,D,x_equi,u_equi,y_equi)
    

 
        
    #Zustandsraumodell des Portals
    def model(self,t,x): 
        # t: Zeit
        # Zustand
        # x[0]: x1 (Relativposition Schlitten in Bezug auf Rahmen)
        # x[1]: x2 (Relativposition Schlitten in Bezug auf Rahmen)
        # x[2]: z1 (Position Rahmen)
        # x[3]: z2 (Position Rahmen)
        # x[4]: phi (Verdrehung Rahmen)
        # x[5]: dx1 (Relativgeschwindigkeit Schlitten in Bezug auf Rahmen)
        # x[6]: dx2 (Relativgeschwindigkeit Schlitten in Bezug auf Rahmen)
        # x[7]: p1 Impuls zu z1
        # x[8]: p2 Impuls zu z2
        # x[9]: phi Drehimpuls zu phi

        _x = self.extract_system_state(x)

        #Eingang auswerten
        u=self._inputs["accell_axis"].input_function(t,x)

        assert(np.shape(u)[0]==(self._inputs["accell_axis"].dimension()))

        #Zustand auspacken
        x1=_x[0]
        x2=_x[1]
        z1=_x[2]
        z2=_x[3]
        phi=_x[4]
        dx1=_x[5]
        dx2=_x[6]
        p1=_x[7]
        p2=_x[8]
        pphi=_x[9]
        ddx1=u[0]
        ddx2=u[1]

        #Winkelfunktionen vorausberechnen
        cphi=np.cos(phi)
        sphi=np.sin(phi)

        ######-------!!!!!!Aufgabe!!!!!!-------------########

        # Lösen des Gleichungssystems mit Hilfe der Massenmatrix
        # Wird für die Zustandsvariablen dx3,dx4,dx5 benötigt
        masssen_matrix = np.array([[self.m+self.M, 0 ,-self.m*(x1*sphi+x2*cphi)],
                                   [0, self.m+self.M, self.m*(x1*cphi-x2*sphi)],
                                   [-self.m*(x1*sphi+x2*cphi),self.m*(x1*cphi-x2*sphi),self.J+self.m*(x1**2+x2**2)]])
        masssen_matrix_inv = np.linalg.inv(masssen_matrix)
        impulse_vektor = np.array([[p1],[p2],[pphi]])
        koordinaten_vektor = np.array([[dx1],[dx2]])
        koordinaten_matrix = np.array([[self.m*cphi, -self.m*sphi],
                                       [self.m*sphi, self.m*cphi],
                                       [-self.m*x2, self.m*x1]])
        dz1,dz2,dphi = np.dot(masssen_matrix_inv,np.dot(-koordinaten_matrix,koordinaten_vektor)+impulse_vektor)
        
        #Hier sollten die korrekten Ableitungen berechnet und zurückgegebenn werden
        dx1=dx1
        dx2=dx2
        dz1=dz1
        dz2=dz2
        dphi=dphi
        dp1=-4*self.k*z1
        dp2=-4*self.k*z2
        dpphi=(-4*self.k*self.L**2*sphi + 
               self.m*(-dphi*dz1*x1-dphi*dz2*x2-dz1*dx2+dx1*dz2)*cphi +
               self.m*(dphi*dz1*x2-dphi*dz2*x1-dz1*dx1+dz2*dx1)*sphi)

        dx=np.array([dx1,dx2,dz1,dz2,dphi,ddx1,ddx2,dp1,dp2,dpphi])

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return dx

    def _gyro_frame_scalar(self,t,x):
        _x = self.extract_system_state(x)
        
        assert(_x.ndim==1)
        #Zustand auspacken
        x1=_x[0]
        x2=_x[1]
        z1=_x[2]
        z2=_x[3]
        phi=_x[4]
        dx1=_x[5]
        dx2=_x[6]
        p1=_x[7]
        p2=_x[8]
        pphi=_x[9]

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekte Ausgangsgleichung implementiert werden
        dphi=np.zeros((1,))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return dphi

    def _accell_frame_scalar(self,t,x):
        #Zustand auspacken
        _x = self.extract_system_state(x)
        assert(_x.ndim==1)
        x1=_x[0]
        x2=_x[1]
        z1=_x[2]
        z2=_x[3]
        phi=_x[4]
        dx1=_x[5]
        dx2=_x[6]
        p1=_x[7]
        p2=_x[8]
        pphi=_x[9]

        #Winkelfunktionen vorausberechnen
        cphi=np.cos(phi)
        sphi=np.sin(phi)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekte Ausgangsgleichung implementiert werden
        ddz=np.zeros((2,))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
       
        return ddz

    def accell_frame(self,t,x):
        _x = self.extract_system_state(x)
        if _x.ndim==1:
            result=self._accell_frame_scalar(t,x)
        else:
            result=np.zeros((2,_x.shape[1]))
            x_scalar={}
            for ii in range(_x.shape[1]):
                if isinstance(x, dict):
                    for sys_name in x.keys():
                        x_scalar[sys_name]=x[sys_name][:,ii]
                else:
                    x_scalar=x[:,ii]
                result[:,ii]=self._accell_frame_scalar(t[ii],x_scalar)
        return result        
    
    def gyro_frame(self,t,x):
        _x = self.extract_system_state(x)
        if _x.ndim==1:
            result=self._gyro_frame_scalar(t,x)
        else:
            result=np.zeros((1,_x.shape[1]))
            for ii in range(x.shape[1]):
                result[:,ii]=self._gyro_frame_scalar(t[ii],x[:,ii])
        return result        


    def position_total(self,t,x):
        _x = self.extract_system_state(x)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekte Ausgangsgleichung implementiert werden
        if _x.ndim==1:
            y1 = _x[2]+np.cos(_x[4])*_x[0]-np.sin(_x[4])*_x[1]
            y2 = _x[3]+np.sin(_x[4])*_x[0]+np.cos(_x[4])*_x[1]
            y = np.array([y1,y2])
            return y
        else:
            return np.zeros_like(_x)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def position_axis(self,t,x):
        _x = self.extract_system_state(x)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekte Ausgangsgleichung implementiert werden
        if _x.ndim==1:
            y1 = _x[0]
            y2 = _x[1]
            y = np.array([y1,y2])
            return y  
        else:
            return np.zeros_like(_x)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
            
    def number_of_states(self):
        return 10;
    

    def verify_model(self,filename):
        import pickle
        with open(filename,'rb') as f:
            x, u, dx_load, y_load = pickle.load(f)
        f.close()
        dx=np.zeros_like(x)
        y=np.zeros_like(u)
        count=dx.shape[1]
        tmp=self._inputs["accell_axis"].input_function
        self._inputs["accell_axis"].input_function = lambda t,x: u[:,0]
        y={}
        for output_name,output_function in self._outputs.items():
            h=output_function(0,x[:,0])
            y[output_name]=np.zeros((h.shape[0],count))
        for ii in range(count):
            self._inputs["accell_axis"].input_function = lambda t,x: u[:,ii]
            dx[:,ii]=self.model(None,x[:,ii])
            for output_name,output_function in self._outputs.items():
                y[output_name][:,ii]=output_function(None,x[:,ii])
        self._inputs["accell_axis"].input_function= tmp
        error_dx_abs_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0))
        error_dx_rel_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0)/np.linalg.norm(dx_load,2,axis=0))
        print("Maximaler absoluter Fehler in Modellgleichung (euklidische Norm):",error_dx_abs_max)
        print("Maximaler relativer Fehler in Modellgleichung (euklidische Norm):",error_dx_rel_max)
        for output_name,_y in y.items():
            error_y_abs_max=np.max(np.linalg.norm(_y-y_load[output_name],2,axis=0))
            error_y_rel_max=np.max(np.linalg.norm(_y-y_load[output_name],2,axis=0)/np.linalg.norm(y_load[output_name],2,axis=0))
            print("Maximaler absoluter Fehler in Ausgang " + output_name + " (euklidische Norm):",error_y_abs_max)
            print("Maximaler relativer Fehler in Ausgang " + output_name + " (euklidische Norm):",error_y_rel_max)
        dx_load_max=np.max(np.linalg.norm(dx_load,2,axis=0))
        y_load_max=np.max(np.linalg.norm(y_load[output_name],2,axis=0))
        



#Trajektorie realisiert polynomialem Übergang zwischen Ruhelagen ohne Berücksichtigung der Elastizitäten 
#t: Auswertezeitpunkte
#ya, yb: Anfangs- und Endwerte für die Schlittenposition jeweils als array
#T: Übergangszeit
#N: Differenzierbarkeit der Solltrajektorie für Position
class RigidTrajectory(DynamicSystem):
    def __init__(self,ya,yb,T,maxderi=None):
        super().__init__()
        self.T=T
        self.ya=ya
        self.yb=yb
        if maxderi==None:
            self.maxderi=[2,2]
        else:
            self.maxderi=maxderi
        self._outputs["u_ref"]=lambda t,x:self.input(t)
        self._outputs["x_ref"]=lambda t,x:self.state(t)
        self._outputs["y_ref"]=lambda t,x:self.output(t)
            
    # Trajektorie des Ausgangs
    def output(self,t,index,derivative):
        tau = t  / self.T
        if derivative==0:
            result = self.ya[index] + (self.yb[index] - self.ya[index]) * poly_transition(tau,0,self.maxderi[index])
        else:
            result = (self.yb[index] - self.ya[index]) * poly_transition(tau,derivative,self.maxderi[index])/self.T**derivative
        return result

    #Zustandstrajektorie 
    def state(self,t):
        tv=np.atleast_1d(t)
        dim_u=2
        dim_x=4
        result = np.vstack([self.output(tv,0,0),self.output(tv,1,0),self.output(tv,0,1),self.output(tv,1,1)])
        if (np.isscalar(t)):
            result=result[:,0]
        return result

        
    #Eingangstrajektorie
    def input(self,t):
        tv=np.atleast_1d(t)
        dim_u=2
        dim_x=2
        result = np.vstack([self.output(tv,0,2),self.output(tv,1,2)])
        if (np.isscalar(t)):
            result=result[:,0]
        return result
   
   


def plot_results(t,x,u,y):
    plt.figure(figsize=(15,7))
    plt.subplot(2,3,1,ylabel="Winkel $\\varphi$ in Grad")
    plt.grid()
    leg=["Soll","Ist"]
    for v in x:
        plt.plot(t,v[4,:]/np.pi*180)
    plt.legend(leg)
    plt.subplot(2,3,2,ylabel="Positionen $z_1$ und $z_2$ in cm")
    plt.grid()
    for v in x:
        plt.plot(t,v[2,:]*100)
        plt.plot(t,v[3,:]*100)
    plt.legend(["Soll $z_1$","Soll $z_2$","Ist $z_1$","Ist $z_2$"])
    plt.subplot(2,3,4,ylabel="Eingang 1 m/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[0,:])
    plt.legend(leg)
    plt.subplot(2,3,5,ylabel="Eingang 2 m/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[1,:])
    plt.legend(leg)
    plt.subplot(2,3,3,ylabel="Positionen $y_1$ in cm")
    plt.grid()
    #leg=["Soll","Ist"]
    #for v in x:
    if len(y)>1:
        #plt.plot(t,(x[1][1,:]-x[0][1,:])*100)
        plt.plot(t,(y[0][0,:])*100)
        plt.plot(t,(y[1][0,:])*100)
    else:
        plt.plot(t,y[0][0,:]*100)
    plt.legend(leg)
    plt.subplot(2,3,6,ylabel="Positionen $y_2$ in cm")
    plt.grid()
    leg=["Soll","Ist"]
    if len(y)>1:
        #plt.plot(t,(x[1][1,:]-x[0][1,:])*100)
        plt.plot(t,(y[0][1,:])*100)
        plt.plot(t,(y[1][1,:])*100)
    else:
        plt.plot(t,y[0][1,:]*100)
    plt.legend(leg)

class GantryObserverModel(DynamicSystem):
    def __init__(self,gantry):
        super().__init__()
        self.L1=gantry.L1
        self.L2=gantry.L2
        self.L=gantry.L
        self.M=gantry.M
        self.J=gantry.J
        self.m=gantry.m
        self.k=gantry.k
        self._inputs["accell_frame"]=InputConnector(2)
        self._outputs["gyro_frame"]=self.gyro_frame
        self._outputs["position_axis"]=self.position_axis
        self._outputs["position_total"]=self.position_total
        self._outputs["state"]=lambda t,x: self.extract_system_state(x)
  
    def number_of_states(self):
        return 10
    
    def position_total(self,t,x):
        _x = self.extract_system_state(x)
        if _x.ndim==1:
            return np.array([_x[0]*np.cos(_x[4])-_x[1]*np.sin(_x[4])+_x[2],
                             _x[0]*np.sin(_x[4])+_x[1]*np.cos(_x[4])+_x[3]])
        else:
            return np.vstack([_x[0,:]*np.cos(_x[4,:])-_x[1,:]*np.sin(_x[4,:])+_x[2,:],
                              _x[0,:]*np.sin(_x[4,:])+_x[1,:]*np.cos(_x[4,:])+_x[3,:]])

    def position_axis(self,t,x):
        _x = self.extract_system_state(x)
        if _x.ndim==1:
            return np.array([_x[0], _x[1]])
        else:
            return np.arry([_x[0,:], _x[1,:]])

    def gyro_frame(self,t,x):
        _x = self.extract_system_state(x)
        if _x.ndim==1:
            return np.array([_x[9:]])
        else:
            return np.array([_x[9:,:]])

    def model(self,t,x):
        _x = self.extract_system_state(x)
        assert(_x.ndim==1)

        #Zustand auspacken
        x1=_x[0]
        x2=_x[1]
        z1=_x[2]
        z2=_x[3]
        phi=_x[4]
        dx1=_x[5]
        dx2=_x[6]
        dz1=_x[7]
        dz2=_x[8]
        dphi=_x[9]

        #Eingang auswerten
        u=self._inputs["accell_frame"].input_function(t,x)
        assert(np.shape(u)[0]==self._inputs["accell_frame"].dimension())

        #Winkelfunktionen vorausberechnen
        cphi=np.cos(phi)
        sphi=np.sin(phi)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Ableitungen berechnet und zurückgegebenn werden
        dx1=0
        dx2=0
        dz1=0
        dz2=0
        dphi=0
        ddx1=0
        ddx2=0
        ddz1=0
        ddz2=0
        ddphi=0

        dx=np.array([dx1,dx2,dz1,dz2,dphi,ddx1,ddx2,ddx1,ddx2,ddphi])

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        dx=np.array([dx1,dx2,dz1,dz2,dphi,ddx1,ddx2,ddz1,ddz2,ddphi])
        return dx 

    def linearize(self,equi,debug=False):
        #Koordinaten der Ruhelage
        x1=equi[0]
        x2=equi[1]

        #Ruhlagen in Variablen
        x_equi=np.zeros((self.number_of_states(),))
        x_equi[0]=x1
        x_equi[1]=x2
        u_equi={'accell_frame':np.zeros((2,))}
        y_equi={'position_total':x_equi[0:2],
                'position_axis':x_equi[0:2],
                'gyro_frame':np.zeros((1,)),
                'state':x_equi}

        
        A=np.zeros((self.number_of_states(),self.number_of_states()))

        B={'accell_frame':np.zeros((self.number_of_states(),2))};

        C={'position_total':np.zeros((2,self.number_of_states())),
           'position_axis':np.zeros((2,self.number_of_states())),
           'gyro_frame':np.zeros((1,self.number_of_states())),
           'state':np.eye(self.number_of_states())}

        D={'position_total':{'accell_frame':np.zeros((2,2))},
           'position_axis':{'accell_frame':np.zeros((2,2))},
           'gyro_frame':{'accell_frame':np.zeros((1,2))},
           'state':{'accell_frame':np.zeros((self.number_of_states(),2))}}

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier die sollten die korrekten Matrizen angegeben werden
        A=np.zeros((10,10))
        B['accell_frame']=np.zeros((10,2))
        C['position_total']=np.zeros((2,10))
        C['position_axis']=np.zeros((2,10))
        C['gyro_frame']=np.zeros((1,10))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        D['position_total']['accell_axis']=np.zeros((2,2))
        D['position_axis']['accell_axis']=np.zeros((2,2))
        D['gyro_frame']['accell_axis']=np.zeros((1,2))

        return ContinuousLinearizedSystem(A,B,C,D,x_equi,u_equi,y_equi)


class GantryObserver(GantryObserverModel):
    def __init__(self,gantry):
        super().__init__(gantry)
        self._inputs["position_axis_sys"]=InputConnector(2)
        self._inputs["gyro_frame_sys"]=InputConnector(1)
    
        self.gain=np.zeros((10,3))
        

    def set_gain(self,gain):
        self.gain=gain

    def model(self,t,x):
        _x = self.extract_system_state(x)

        #Eingang auswerten
        y_axis_sys=self._inputs["position_axis_sys"].input_function(t,x)
        y_gyro_sys=self._inputs["gyro_frame_sys"].input_function(t,x)
        assert(np.shape(y_axis_sys)[0]==self._inputs["position_axis_sys"].dimension())
        assert(np.shape(y_gyro_sys)[0]==self._inputs["gyro_frame_sys"].dimension())
        y_sys=np.hstack((y_axis_sys,y_gyro_sys))
        y=np.hstack((_x[0:2],_x[9:]))


        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Bitte anpassen
        dx = super().model(t,x)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        return dx
        
