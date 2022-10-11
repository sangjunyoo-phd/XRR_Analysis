"""
X-ray reflectivity Module
Only applies for E != E_edge: mu=0
Goals
1. Object-Oriented for a better manipulation
2. Vizualization of the slab model EDP, step EDP, and R/RF
3. Parameter Estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from math import erf
from matplotlib import rcParams
plt.rcParams['font.size'] = 8
rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
from lmfit import minimize, Parameters, report_fit, Parameter

class Layer:
    def __init__(self, thickness, rho, sigma):
        self.thickness = thickness # Thickness of a slab
        self.rho = rho # Electron density of a slab
        self.sigma = sigma # Roughness of a slab (top: between i and i+1 layers)
        
class Model:
    def __init__(self, substrate='Water', superphase='Air'):
        """
        Default: Air-Water interface
        Initialized with substrate layer with roughness=3
        Add more default substrates and superphases when needed
        """
        self.params = Parameters()
        self.lkey = [0,1] # A key to access a layer
        if substrate in ['Water', 'water', 'H2O', 'h2o']:
            self.params.add('_d'+str(self.lkey[0]), value=1.0, vary=False)
            self.params.add('_rho'+str(self.lkey[0]), value=0.333, vary=False)
            self.params.add('_sig'+str(self.lkey[0]), value=3.0, vary=True, min=1.5, max=8)
            
        if superphase in ['Air', 'air', 'He', 'he', 'Helium', 'helium', 'vacuum']:
            self.params.add('_d'+str(self.lkey[-1]), value=0.0, vary=False)
            self.params.add('_rho'+str(self.lkey[-1]), value=0.0, vary=False)
            self.params.add('_sig'+str(self.lkey[-1]), value=0.0, vary=False)

    def add(self, Layer):
        """
        Adding a Layer to the model
        1. Append lkey list (0: water -1: air)
        2. Add the input layer & input layer parameter
        """
        self.lkey.append(int(self.lkey[-1]+1))
        
        # Parameters of the added layer
        layer_par = [Layer.thickness, Layer.rho, Layer.sigma]
        
        for i, par in enumerate(['_d','_rho','_sig']):
            # Superphase
            self.params[f'{par}{self.lkey[-1]}']=self.params[f'{par}{self.lkey[-2]}']
            # Added Layer
            self.params.add(f'{par}{self.lkey[-2]}', value=layer_par[i], vary=True, min=0, max=np.inf)
        
    def edp(self):
        """
        Returns z and electron density to plot the electron density profile of the model
        Let the surface of the "SUBSTRATE" located at z=0
        """
        z = np.linspace(-3*self.params['_sig0'],
                        sum(self.params[f'_d{l}'] for l in range(1,self.lkey[-1]))+\
                            3*self.params[f'_sig{self.lkey[-2]}'], num=100)
        #self.params[0,0] = abs(z[0]) # Update the substrate thickness accordingly
        ed = np.ones(shape=z.shape) * self.params['_rho0']

        for l in self.lkey[1:]:
            # l: index of layers except for the substrate
            pos = sum(self.params[f'_d{idx}'] for idx in range(1,l))
            for i in range(len(z)):
                # Update numerical edp value
                ed[i] += 0.5*(self.params[f'_rho{l}'].value-self.params[f'_rho{l-1}'])\
                        *(1+erf((z[i]-pos)/np.sqrt(2)/self.params[f'_sig{l-1}']))
        return z, ed
    
    def stepedp(self):
        """
        Returns d and electron density to plot the step electron density profile of the model
        Let the surface of the "SUBSTRATE" located at z=0
        """
        steped=list()
        z_min = -3*self.params['_sig0']
        
        for l in self.lkey:
            steped.append([z_min,self.params[f'_rho{l}']])
            
            if l==0:
                steped.append([z_min+3*self.params['_sig0'],self.params[f'_rho{l}']])
                z_min = z_min + 3*self.params['_sig0']
            else:
                steped.append([z_min+self.params[f'_d{l}'],self.params[f'_rho{l}']])
                z_min = z_min+self.params[f'_d{l}']
        
        return np.asarray(steped)
    
    def rrf(self, params, q):
        """ Calculates R/RF from parameters saved in the model
        as a function of q """
        #params = self.params
        # read parameter value as val['par_name']
        val = params.valuesdict()
        r = 0j*np.zeros(shape=q.shape) # R/RF = abs(temp)**2        
        for l in self.lkey[:-1]:
            d = sum(val[f'_d{idx}'] for idx in range(l+1,self.lkey[-1]+1))
            r += complex(1.0,0.0)*(val[f'_rho{l}']-val[f'_rho{l+1}'])/val['_rho0']\
                *np.exp(complex(0.0,-1.0)*q*d)\
                *complex(1.0,0.0)*np.exp(-0.5 * q**2 * val[f'_sig{l}']**2)
        rrf = abs(r)**2
        return rrf
    
    def fit(self, exp_data):
        """ Fit the Parameters with lmfit library"""
        data=np.genfromtxt(exp_data)
        q = data[:,0]
        rrf_exp=data[:,1]
        print(self.params.keys())
        
        def fcn2min(params, q, rrf_exp):
            return np.log10(self.rrf(params, q))-np.log10(rrf_exp)
        
        out = minimize(fcn2min, self.params, args=(q,), kws={'rrf_exp': rrf_exp})
        self.params = out.params
        #print(out.params)
        #print(self.params)
        report_fit(out)

    def plot(self, exp_data='None'):
        fig = plt.figure()
        fig_width = 3.25
        aspect_ratio = 1.9
        fig_height = aspect_ratio * fig_width / (1.618)
        fig.set_size_inches(fig_width, fig_height)
        
        # Bottom: EDP and step EDP
        ax1 = fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio])
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left= min(self.edp()[0]),right= 1.05* max(self.edp()[0]))
        ax1.set_xlabel('z ($\AA$)')
        ax1.set_ylabel('Electron Density (e/$\AA^{3}$)')
        ax1.plot(self.edp()[0], self.edp()[1], zorder=1, color='tab:blue')
        ax1.plot(self.stepedp()[:,0],self.stepedp()[:,1],
                 linestyle='--', linewidth=1, color='black',zorder=0)
        # Top: XRR experimental & fitted curve data
        ax2 = fig.add_axes([0.2,1.1/aspect_ratio,0.7,0.7/aspect_ratio])
        ax2.set_yscale('log')
        ax2.set_xlabel('$Q_{z} (\AA^{-1})$')
        ax2.set_ylabel('$R/R_{F}$')
        
        q=np.linspace(0,0.8,500)
        ax2.plot(q, self.rrf(self.params,q), color='k', linewidth=1.5, zorder=3)
        if exp_data != 'None':
            data=np.genfromtxt(exp_data)
            ax2.errorbar(data[:,0],data[:,1],yerr=data[:,2], linewidth=0,
                         marker='o', markersize=5, markerfacecolor='None',
                         markeredgewidth=0.5, markeredgecolor='tab:red',
                         elinewidth=1, capsize=1, color='tab:red')
            
        #Test!!!
        #testdata = np.genfromtxt('test.dat')
        #ax2.plot(testdata[:,0],testdata[:,1], color='tab:red',zorder=1)
            
    def savefig(self, name:str, exp_data='None'):
        self.plot(exp_data)
        plt.savefig(name+'.png', dpi=300)
        

head = Layer(4,0.7,3)
tail = Layer(20,0.3,3)

model = Model()
model.add(head)
model.add(tail)

exp_data='Nd_pure_rrf.txt'
model.fit(exp_data=exp_data)
model.plot(exp_data=exp_data)

#model.savefig('test', 'Nd_pure_rrf.txt')

