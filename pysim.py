import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import mimo

from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

class InputConnector:
    def __init__(self,dimension):
        self._dimension=dimension
        self.input_function=None

    def dimension(self):
        return self._dimension
    
class DynamicSystem:
    def __init__(self):
        self._outputs={};
        self._inputs={};
        self._name=None

    def extract_system_state(self,x):
        if isinstance(x, dict):
            return x[self._name]
        else:
            return x
        
    def set_name(self,name):
        self._name=name
        
    def get_name(self):
        return self._name

    def get_output(self,output_name):
        return self._outputs[outputname]

    def connect(self,input_name,system,output_name):
        assert(input_name in self._inputs)
        assert(output_name in system._outputs)
        self._inputs[input_name].input_function=system._outputs[output_name]
        
    def num_output_ports(self):
        return len(self._outputs)

    def input_names(self):
        return list(self._inputs);

    def number_of_inputs(self,input_name):
        return self._inputs[input_name].dimension();

    def number_of_outputs(self,output_name):
        assert("To be implemented")
    
    def output_names(self):
        return list(self._outputs);
    
    def number_of_states(self):
        return 0

    def num_input_ports(self):
        return len(self._inputs)

    def output(self,t,x,output_name):
        return self._outputs[output_name](t,x)


    def verify_linearization(self,linear_model, eps_x=1e-6,eps_u=1e-6):
        """Compare linear state space model with its approximate Taylor linearization of non-linear model
        Parameters
        ----------
        linear_model : object of type LinearizedModel

        Returns
        -------
        nothing

        Notes
        -----
        Tay

        """
        A_approx=np.zeros_like(linear_model.A())
        B_approx={}
        C_approx={}
        for output_name, function  in self._outputs.items():
            C_approx[output_name]=np.zeros_like(linear_model.C(output_name))

        D_approx={}

        for output_name in self._outputs.keys():
            D_approx[output_name]={}
            for input_name  in self._inputs.keys():
                D_approx[output_name][input_name]=np.zeros_like(linear_model.D(output_name,input_name))

        for input_name in self._inputs.keys():
            B_approx[input_name]=np.zeros_like(linear_model.B(input_name))
                
        x_equi=linear_model.x_equi
        dim_x=linear_model.A().shape[0]
        if np.isscalar(eps_x):
            eps_x=np.ones((dim_x,))*eps_x

        saved_inputs={}
        for input_name,connector in self._inputs.items():
            saved_inputs[input_name]=connector.input_function
            connector.input_function=lambda t,x:linear_model.u_equi[input_name]
                 
        for jj in range(dim_x):
            x_equi1=np.array(x_equi)
            x_equi2=np.array(x_equi)
            x_equi2[jj]+=eps_x[jj]
            x_equi1[jj]-=eps_x[jj]
            #print("Ruhelage:",x_equi)
            dx=(self.model(0,x_equi2)-self.model(0,x_equi1))/2/eps_x[jj]
            A_approx[:,jj]=dx
            for (output_name,function) in self._outputs.items():
                output=self._outputs[output_name]
                dy=(output(0,x_equi2)-output(0,x_equi1))/2/eps_x[jj]
                C_approx[output_name][:,jj]=dy
            
        error_A=np.abs(linear_model.A()-A_approx)
        idx= np.unravel_index(np.argmax(error_A, axis=None), error_A.shape)
        print("Maximaler absoluter Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_A[idx[0],idx[1]])+".")
        scale_A=np.hstack([np.where(abs(A_approx[:,jj:jj+1]) > eps_x[jj], abs(A_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_A=error_A/scale_A
        idx= np.unravel_index(np.argmax(error_rel_A, axis=None), error_rel_A.shape)
        print("Maximaler relativer Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_A[idx[0],idx[1]])+".")

        for output_name in self._outputs.keys():
            error_C=np.abs(linear_model.C(output_name)-C_approx[output_name])
            idx= np.unravel_index(np.argmax(error_C, axis=None), error_C.shape)
            print("Maximaler absoluter Fehler in Matrix C_"+output_name+" Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_C[idx[0],idx[1]])+".")
            scale_C=np.hstack([np.where(abs(C_approx[output_name][:,jj:jj+1]) > eps_x[jj], abs(C_approx[output_name][:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
            error_rel_C=error_C/scale_C
            idx= np.unravel_index(np.argmax(error_rel_C, axis=None), error_rel_C.shape)
            print("Maximaler relativer Fehler in Matrix C_"+output_name+" Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_C[idx[0],idx[1]])+".")

        for (input_name,connector) in self._inputs.items():
            dim_u=connector.dimension()
            if np.isscalar(eps_u):
                eps_u=np.ones((dim_u,))*eps_u
            for jj in range(dim_u):
                url1=np.array(linear_model.u_equi[input_name])
                url2=np.array(linear_model.u_equi[input_name])
                url1[jj]-=eps_u[jj]
                url2[jj]+=eps_u[jj]
                connector.input_function=lambda t,x:url1
                x1=self.model(0,x_equi)
                connector.input_function=lambda t,x:url2
                x2=self.model(0,x_equi)
                dx=(x2-x1)/2/eps_u[jj]
                B_approx[input_name][:,jj]=dx
                for (output_name,output) in self._outputs.items():
                    connector.input_function=lambda t,x:url1
                    y1=output(0,x_equi)
                    connector.input_function=lambda t,x:url2
                    y2=output(0,x_equi)
                    dy=(y2-y1)/2/eps_u[jj]
                    D_approx[output_name][input_name][:,jj]=dy
                connector.input_function=lambda t,x:linear_model.u_equi[input_name]
            
        for (input_name,connector) in self._inputs.items():
            connector.input_function=saved_inputs[input_name]

        for input_name in self._inputs.keys():
            error_B=np.abs(linear_model.B(input_name)-B_approx[input_name])
            idx= np.unravel_index(np.argmax(error_B, axis=None), error_B.shape)
            print("Maximaler absoluter Fehler in Matrix B_"+ input_name+" Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_B[idx[0],idx[1]])+".")
            scale_B=np.hstack([np.where(abs(B_approx[input_name][:,jj:jj+1]) > eps_x[jj], abs(B_approx[input_name][:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
            error_rel_B=error_B/scale_B
            idx= np.unravel_index(np.argmax(error_rel_B, axis=None), error_rel_B.shape)
            print("Maximaler relativer Fehler in Matrix B_"+ input_name+" Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_B[idx[0],idx[1]])+".")

        for output_name in self._outputs.keys():
            for input_name in self._inputs.keys():
                error_D=np.abs(linear_model.D(output_name,input_name)-D_approx[output_name][input_name])
                idx= np.unravel_index(np.argmax(error_D, axis=None), error_D.shape)
                print("Maximaler absoluter Fehler in Matrix D_{"+ output_name+","+input_name +"} Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_D[idx[0],idx[1]])+".")
                scale_D=np.hstack([np.where(abs(D_approx[output_name][input_name][:,jj:jj+1]) > eps_x[jj], abs(D_approx[output_name][input_name][:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
                error_rel_D=error_D/scale_D
                idx= np.unravel_index(np.argmax(error_rel_D, axis=None), error_rel_D.shape)
                print("Maximaler relativer Fehler in Matrix D_{"+ output_name+","+input_name +"} Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_D[idx[0],idx[1]])+".")
        return A_approx, B_approx, C_approx, D_approx

class CompoundSystem(DynamicSystem):
    def __init__(self):
        super().__init__()
        self._systems={}

    def get_system(self,sysname):
        return self._systems[sysname]

    def add_system(self,system,sysname):
        system.set_name(sysname)
        self._systems[sysname]=system

    def assemble_state(self,system_states):
        number_of_states=self.number_of_states()
        #check dimensions
        (sys_name,sys_state)=list(system_states.items())[0]
        ndim=sys_state.ndim
        if ndim==1:
            compound_state=np.zeros((number_of_states,))
        else:
            compound_state=np.zeros((number_of_states,sys_state.shape[1]))
        idx_start=0
        for (sys_name,sys) in list(self._systems.items()):
            if sys_name in system_states:
                if ndim==1:
                    compound_state[idx_start:idx_start+sys.number_of_states()]=system_states[sys_name]
                else:
                    compound_state[idx_start:idx_start+sys.number_of_states(),:]=system_states[sys_name]
            idx_start+=sys.number_of_states()
        return compound_state

    def disassemble_state(self,compound_state):
        number_of_states=self.number_of_states()
        states={}
        idx_start=0
        for (sysname,sys) in self._systems.items():
            if compound_state.ndim==1:
                states[sysname]=compound_state[idx_start:idx_start+sys.number_of_states()]
            else:
                states[sysname]=compound_state[idx_start:idx_start+sys.number_of_states(),:]
            idx_start+=+sys.number_of_states()
        return states
        
    def number_of_states(self):
        number_of_states=0
        for key in self._systems.keys():
            number_of_states+=self._systems[key].number_of_states()
        return number_of_states
    
    def model(self,t,x):
        idx_start=0
        dx_={}
        x_=self.disassemble_state(x)
        for (sysname,sys) in self._systems.items():
            if sys.number_of_states()>0:
                dx_[sysname]=sys.model(t,x_)
            idx_start+=sys.number_of_states()
        dx=self.assemble_state(dx_)
        return dx
  
            

                                                                     

class LinearizedSystem(DynamicSystem):
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        super().__init__()
        self._A=A
        self._B=B
        self._C=C
        self._D=D
        self.x_equi=x_equi
        self.u_equi=u_equi
        self.y_equi=y_equi
        for key in B.keys():
            self._inputs[key]=InputConnector(B[key].shape[0])
        for key in C.keys():
            self._outputs[key]=lambda t,x:self.output(t,x,key)

    def B(self,input_name=None):
        if input_name==None:
            assert(self.num_input_ports()==1)
            input_name=self.input_names()[0];
        assert(input_name in self._inputs)            
        return self._B[input_name]

    def A(self):
        return self._A
    
    def C(self,output_name=None):
        if output_name==None:
            assert(self.num_output_ports()==1)
            output_name=self.output_names()[0];
        assert(output_name in self._outputs)            
        return self._C[output_name]

    def D(self,output_name=None,input_name=None):
        if output_name==None:
            assert(self.num_output_ports()==1)
            output_name=self.output_names()[0];
        if input_name==None:
            assert(self.num_input_ports()==1)
            input_name=self.input_names()[0];
        assert(output_name in self._outputs)            
        assert(input_name in self._inputs)            
        return self._D[output_name][input_name]

    #Ackermannformel
    def acker(self,eigs,input_name=None):
        return mimo.acker(self.A(),self.B(input_name),eigs)

    #Berechnung eines beliebigen Ausgangs
    def output(self,t,x,output_name):
        _x = self.extract_system_state(x)
        if x.ndim==1:
            #y = self.C@(x-self.x_equi) + self.y_equi # + self.D@(u-self.u_equi)
            x_equi=self.x_equi
            y_equi=self.y_equi[output_name]
        else:
            x_equi=self.x_equi.reshape((self.x_equi.shape[0],1))
            y_equi=self.y_equi[output_name].reshape((self.y_equi[output_name].shape[0],1))

        
        #Zustandsabhängigkeit
        y=self.C(output_name)@(_x-x_equi)
        
        #Eingänge auswerten
        for input_name, _D in self._D[output_name].items():
            if x.ndim==1:
                u_equi=self.u_equi[input_name]
            else:
                u_equi=self.u_equi[input_name].reshape((self.u_equi[input_name].shape[0],1))
            #Eingang nur auswerden, wenn tatsächlich ein Durchgriff existiert,
            #um algebraische Schleifen möglichst zu vermeiden
            if _D.any():
                y+=_D@(self._inputs[input_name].input_function(t,x)-u_equi)

        #Arbeitspunkt aufschalten
        y=y+y_equi

        return y

class ContinuousLinearizedSystem(LinearizedSystem):
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        super().__init__(A,B,C,D,x_equi,u_equi,y_equi)
        
    def discretize(self,Ta):
        assert Ta>0
        Ad, Bd, Cd, Dd, Ta = cont2discrete((self.A, self.B , self.C, self.D),Ta)
        return DiscreteLinearizedSystem(Ad,Bd,Cd,Dd,self.x_equi,self.u_equi,self.y_equi,Ta)

    #Quadratisch optimaler Regler
    def lqr(self,Q,R,input_name=None):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr=np.zeros((R.shape[0],Q.shape[0]))
        return K_lqr
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def rest_to_rest_trajectory(self,equis,T,kronecker,maxderi=None, input_name=None):
        return ContinuousFlatnessBasedTrajectory(self,equis,T,kronecker,maxderi, input_name)
        
    #linearisiertes Zustandsraumodell des Systems
    def model(self,t,x): 
        # t: Zeit
        # x: Zustand (entweder 

        _x = self.extract_system_state(x)
        x_equi=self.x_equi

        #Zustandsabhängigkeit
        dx=self._A@(_x-self.x_equi)
        
        #Eingang auswerten 
        for input_name, _B in self._B.items():
            u_equi=self.u_equi[input_name]
            dx+=_B@(self._inputs[input_name].input_function(t,x)-u_equi)

        #Ableitungen zurückgeben
        return dx

class ContinuousFlatnessBasedTrajectory(DynamicSystem):
    """Zeitkontinuierliche flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für das lineare zeitkontinuierliche Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitete worden sind.

    Args:
       equis (dict): dict Dictionary mit  Anfangs- und Endwerten den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximal Differenzierbarkeitsanforderungen für flachen Ausgang (None entspricht maxderi=kronecker)
       input: Name des Eingangs der für die Regelungsnormalform genutzt wird
    """
    def __init__(self,linearized_system,equis,T,kronecker,maxderi=None, input_name=None):
        super().__init__()
        if input_name==None:
            assert(linearized_system.num_input_ports()==1)
            input_name=linearized_system.input_names()[0];

        self.input_name=input_name
        self.linearized_system=linearized_system
        self.T=T

        #Anfangs und Endbedingungen zur Parametrierung holen und auf den Arbeitspunkt beziehen
        output_name, _equis=list(equis.items())[0]
        
        y_start_rel = _equis["start"]-linearized_system.y_equi[output_name]
        y_final_rel = _equis["final"]-linearized_system.y_equi[output_name]
        
        #Testen ob Ausgang existiert
        assert(output_name in linearized_system.output_names())

        #Ausgangs-Matrix für die Parametrierung
        C=linearized_system.C(output_name)
        
        self.kronecker=np.array(kronecker,dtype=int)
        if maxderi==None:
            self.maxderi=self.kronecker
        else:
            self.maxderi=self.maxderi
        
        #Matrizen der Regelungsnormalform holen
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier bitte benötigte Zeilen wieder "dekommentieren" und Rest löschen
        self.A_rnf, Brnf, Crnf, self.M, self.Q, S = mimo.rnf(linearized_system.A(),
                                                            linearized_system.B(input_name),
                                                             linearized_system.C(output_name),
                                                             kronecker)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        self._outputs["u_ref"]=lambda t,x:self.input(t)
        self._outputs["x_ref"]=lambda t,x:self.state(t)
        for output in linearized_system.output_names():
            self._outputs[output_name+"_ref"]=lambda t,x:self.output(t,output_name)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang
        #Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        #Achtung: Hier sollten alle werte relativ zum Arbeitspunkt angegeben werden
        # y_start_rel Anfangswerte des vorgegebenen Ausgangs
        # y_final_rel Endwerte des vorgegebenen Ausgangs
        # Crnf ... Ausgangsmatrix in Regelungsnormalform zum vorgegebnen Ausgang
        
        Crnf_equi = Crnf[:,(0,kronecker[0])]
        Crnf_equi_inv = np.linalg.inv(Crnf_equi)

        self.eta_start=Crnf_equi_inv@y_start_rel
        self.eta_final=Crnf_equi_inv@y_final_rel

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    # Trajektorie des flachen Ausgangs
    def flat_output(self,t,index,derivative):
        
        tau = t  / self.T
        if derivative==0:
            return self.eta_start[index] + (self.eta_final[index] - self.eta_start[index]) * mimo.poly_transition(tau,0,self.maxderi[index])
        else:
            return (self.eta_final[index] - self.eta_start[index]) * mimo.poly_transition(tau,derivative,self.maxderi[index])/self.T**derivative 

    #Zustandstrajektorie 
    def state(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi[self.input_name])
        dim_x=np.size(self.linearized_system.x_equi)
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(tv,index,deri) for deri in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        result = np.linalg.inv(self.Q)@xrnf+self.linearized_system.x_equi.reshape((dim_x,1))
        if (np.isscalar(t)):
            result=result[:,0]
        return result


    #Ausgangstrajektorie für eine beliebigen Ausgang
    def output(self,t,output_name):
        y_equi=self.linearized_system.y_equi[output_name]
        x_equi=self.linearized_system.x_equi
        dim_y=np.size(y_equi)
        dim_x=np.size(x_equi)
        x_abs=self.state(t)
        x_rel=x_abs-x_equi.reshape((dim_x,1))
        y_rel=self.linearized_system.C(output_name)@x_rel
        y_abs=y_rel+y_equi.reshape((dim_y,1))
        if (np.isscalar(t)):
            y_abs=result[:,0]
        return y_abs

    #Eingangstrajektorie
    def input(self,t):
        tv=np.atleast_1d(t)
        u_equi=self.linearized_system.u_equi[self.input_name]
        dim_u=np.size(u_equi)
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(tv,index,deri) for deri in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        v=-self.A_rnf[self.kronecker.cumsum()-1,:]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj,:]+=self.flat_output(tv,jj,self.kronecker[jj])
        result = (np.linalg.inv(self.M)@v)+u_equi.reshape((dim_u,1))
        if (np.isscalar(t)):
            result=result[:,0]
        return result




class LinearStateFeedback(DynamicSystem):
    def __init__(self,gain):
        super().__init__()
        self._outputs['u']=self.output

        #Eingänge anlegen
        self._inputs['u_ref']=InputConnector(gain.shape[0])
        self._inputs['x_ref']=InputConnector(gain.shape[1])
        self._inputs['x']=InputConnector(gain.shape[1])
        self._gain=gain

        #Ausgang registrieren
        self._outputs['u']=self.output

    def output(self,t,x):
        #Sollstellgröße
        u_ref=self._inputs['u_ref'].input_function(t,x)

        #Sollzustand
        x_ref=self._inputs['x_ref'].input_function(t,x)

        #aktueller Systemzustand
        x=self._inputs['x'].input_function(t,x)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollt das korrekte Regelgesetz angeben implementiert werden
        #Reglerverstärkung ist self._gain

        u=np.zeros_like(u_ref)

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        
        return u
        
    def set_gain(self,gain):
        self._gain=gain
