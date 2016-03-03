import numpy as np
import scipy.sparse
from numpy import sin
from numpy import cos
import sys
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
opts = PETSc.Options()
try:
    import cPickle as pkl
except:
    import pickle as pkl

class Model():
    def __init__(self, model_variables, boundary_variables,
                 model_parameters, physical_constants):
        self.model_variables = model_variables
        self.boundary_variables = boundary_variables
        self.model_parameters = model_parameters
        self.physical_constants = physical_constants

        for key in model_parameters:
            exec('self.'+str(key)+' = model_parameters[\''+str(key)+'\']')
        for key in physical_constants:
            exec('self.'+str(key)+' = physical_constants[\''+str(key)+'\']')

        self.calculate_nondimensional_parameters()
        self.set_up_grid(self.R, self.h)
        
    def set_up_grid(self, R, h):
        '''
        Creates the r and theta coordinate vectors
        inputs:
            R: radius of outer core in m
            h: layer thickness in m
        outputs: None
        '''
        self.R = R
        self.h = h
        self.Size_var = self.Nk*self.Nl
        self.SizeMnoBC = len(self.model_variables)*self.Size_var
        self.SizeM = self.SizeMnoBC + 2*len(self.boundary_variables)*self.Nl
        self.rmin = (self.R-self.h)/self.r_star
        self.rmax = self.R/self.r_star
        self.dr = (self.rmax-self.rmin)/(self.Nk)
        self.r = np.linspace(self.rmin-self.dr/2., self.rmax+self.dr/2.,num=self.Nk+2.) # r value at center of each cell
        self.rm = np.linspace(self.rmin-self.dr, self.rmax, num=self.Nk+2.) # r value at plus border (top) of cell
        self.rp = np.linspace(self.rmin, self.rmax+self.dr, num=self.Nk+2.) # r value at minus border (bottom) of cell
        self.dth = np.pi/(self.Nl)
        self.th = np.linspace(-self.dth/2., np.pi+self.dth/2., num=self.Nl+2.) # theta value at center of cell
        self.thm = np.linspace(-self.dth, np.pi, num=self.Nl+2.) # theta value at plus border (top) of cell
        self.thp = np.linspace(0,np.pi+self.dth, num=self.Nl+2.)
        return None

    def calculate_nondimensional_parameters(self):
        '''
        Calculates the non-dimensional parameters in model from the physical
        constants.
        '''
        self.t_star = 1/self.Omega  # seconds
        self.r_star = self.R  # meters
        self.P_star = self.rho*self.r_star**2/self.t_star**2
        self.B_star = (self.eta*self.mu_0*self.rho/self.t_star)**0.5
        self.u_star = self.r_star/self.t_star
        self.E = self.nu*self.t_star/self.r_star**2
        self.Prm = self.nu/self.eta
        return None

    def set_Br(self, BrT):
        ''' Sets the backgound magnetic field in Tesla
        BrT = Br values for each cell in Tesla'''
        self.BrT = BrT
        self.Br = self.BrT/self.B_star
        return None

    def set_Bth(self, BthT):
        ''' Sets the background theta magnetic field in Tesla
        BthT = Bth values for each cell in Tesla'''
        self.BthT = BthT
        self.Bth = self.BthT/self.B_star

    def set_Br_dipole(self, Bd, const=0):
        ''' Sets the background magnetic field to a dipole field with
        Bd = dipole constant in Tesla '''
        self.Bd = Bd
        self.BrT = 2*np.ones((self.Nk+2, self.Nl+2))*cos(self.th)*Bd + const
        self.Br = self.BrT/self.B_star
        self.BthT = np.ones((self.Nk+2, self.Nl+2))*0.0
        self.Bth = self.BthT/self.B_star
        return None

    def set_B_dipole(self, Bd, const=0):
        ''' Sets the background magnetic field to a dipole field with
        Bd = dipole constant in Tesla '''
        self.Bd = Bd
        self.BrT = 2*np.ones((self.Nk+2, self.Nl+2))*cos(self.th)*Bd + const
        self.Br = self.BrT/self.B_star
        self.BthT = np.ones((self.Nk+2, self.Nl+2))*sin(self.th)*Bd + const
        self.Bth = self.BthT/self.B_star
        return None

    def set_B_abs_dipole(self, Bd, const=0):
        ''' Sets the background magnetic Br and Bth field to the absolute value of a
        dipole field with Bd = dipole constant in Tesla '''
        self.Bd = Bd
        self.BrT = 2*np.ones((self.Nk+2, self.Nl+2))*abs(cos(self.th))*Bd + const
        self.Br = self.BrT/self.B_star
        self.BthT = np.ones((self.Nk+2, self.Nl+2))*abs(sin(self.th))*Bd + const
        self.Bth = self.BthT/self.B_star
        return None

    def set_B_dipole_absrsymth(self, Bd, const=0):
        ''' Sets the background magnetic Br and Bth field to the absolute value of a
        dipole field with Bd = dipole constant in Tesla '''
        self.Bd = Bd
        self.BrT = 2*np.ones((self.Nk+2, self.Nl+2))*abs(cos(self.th))*Bd + const
        self.Br = self.BrT/self.B_star
        self.BthT = np.ones((self.Nk+2, self.Nl+2))*sin(self.th)*Bd + const
        self.Bth = self.BthT/self.B_star
        return None

    def set_Br_abs_dipole(self, Bd, const=0, noise=None, N=10000):
        ''' Sets the background Br magnetic field the absolute value of a
        dipole with Bd = dipole constant in Tesla.
        optionally, can offset the dipole by a constant with const or add numerical noise with noise '''
        if noise:     
            from scipy.special import erf
            def folded_mean(mu, s):
                return s*(2/np.pi)**0.5*np.exp(-mu**2/(2*s**2)) - mu*erf(-mu/(2*s**2)**0.5)
            self.Bd = Bd
            Bdip = 2*Bd*np.abs(np.cos(self.th))
            Bdip_noise = np.zeros_like(Bdip)
            for (i,B) in enumerate(Bdip):
                Bdip_noise[i] = folded_mean(Bdip[i], noise)
            self.BrT = np.ones((self.Nk+2, self.Nl+2))*Bdip_noise
            self.Br = self.BrT/self.B_star
            self.BthT = np.ones((self.Nk+2, self.Nl+2))*0.0
            self.Bth = self.BthT/self.B_star
        else:
            self.Bd = Bd
            self.BrT = 2*np.ones((self.Nk+2, self.Nl+2))*abs(cos(self.th))*Bd + const
            self.Br = self.BrT/self.B_star
            self.BthT = np.ones((self.Nk+2, self.Nl+2))*0.0
            self.Bth = self.BthT/self.B_star
        return None

    def set_Br_sinfunc(self, Bmin, Bmax, sin_exp=2.5):
        self.BrT = np.ones((self.Nk+2, self.Nl+2))*((1-sin(self.th)**sin_exp)*(Bmax-Bmin)+Bmin)
        self.Br = self.BrT/self.B_star
        self.BthT = np.ones((self.Nk+2, self.Nl+2))*0.0
        self.Bth = self.BthT/self.B_star
        return None
        
        
    def set_B_by_type(self, B_type, Bd=0.0, Br=0.0, Bth=0.0, const=0.0, Bmin=0.0, Bmax=0.0, sin_exp=2.5, noise=0.0):
        ''' Sets the background magnetic field to given type.
        B_type choices:
            * dipole : Br, Bth dipole; specify scalar dipole constant Bd (T)
            * abs_dipole : absolute value of dipole in Br and Bth, specify scalar Bd (T)
            * dipole_Br : Br dipole, Bth=0; specify scalar dipole constant Bd (T)
            * abs_dipole_Br : absolute value of dipole in Br, specify scalar Bd (T)
            * constant_Br : constant Br, Bth=0; specify scalar Br (T)
            * set : specify array Br and Bth values in (T)
            * dipole_absrsymth : absolute value of dipole in Br, symmetric in Bth, specify scalar Bd (T)
        '''
        if B_type == 'dipole':
            self.set_B_dipole(Bd, const=const)
        elif B_type == 'dipoleBr':
            self.set_Br_dipole(Bd, const=const)
        elif B_type == 'constantBr':
            self.set_Br(Br*np.ones((self.Nk+2, self.Nl+2)))
            self.set_Bth(0.0*np.ones((self.Nk+2, self.Nl+2)))
        elif B_type == 'set':
            self.set_Br(Br)
            self.set_Bth(Bth)
        elif B_type == 'absDipoleBr':
            self.set_Br_abs_dipole(Bd, const=const, noise=noise)
        elif B_type == 'absDipole':
            self.set_B_abs_dipole(Bd, const=const)
        elif B_type == 'dipoleAbsRSymTh':
            self.set_B_dipole_absrsymth(Bd, const=const)
        elif B_type == 'sinfuncBr':
            self.set_Br_sinfunc(Bmin, Bmax, sin_exp=sin_exp)
        else:
            raise Exception('B_type not valid')

    def set_CC_skin_depth(self, period):
        ''' sets the magnetic skin depth for conducting core BC
        inputs:
            period = period of oscillation in years
        returns:
            delta_C = skin depth in (m)
        '''
        self.delta_C = np.sqrt(2*self.eta/(2*np.pi/(period*365.25*24*3600)))
        self.physical_constants['delta_C'] = self.delta_C
        return self.delta_C

    def set_Uphi(self, Uphi):
        '''Sets the background velocity field in m/s'''
        self.Uphi = Uphi
        self.U0 = self.Uphi*self.r_star/self.t_star
        return None

    def set_buoyancy(self, drho_dr):
        '''Sets the buoyancy structure of the layer'''
        self.omega_g = np.sqrt(-self.g/self.rho*drho_dr)
        self.G = self.omega_g**2*self.t_star**2

    def set_buoy_by_type(self, buoy_type, buoy_ratio):
        self.omega_g0 = buoy_ratio*self.Omega
        if buoy_type == 'constant':
            self.omega_g = np.ones((self.Nk+2, self.Nl+2))*self.omega_g0
        elif buoy_type == 'linear':
            self.omega_g = (np.ones((self.Nk+2, self.Nl+2)).T*np.linspace(0, self.omega_g0, self.Nk+2)).T
        self.G = self.omega_g**2*self.t_star**2

    def get_index(self, k, l, var):
        '''
        Takes coordinates for a point, gives back index in matrix.
        inputs:
            k: k grid value from 1 to K (or 0 to K+1 for variables with
                boundary conditions)
            l: l grid value from 1 to L
            var: variable name in model_variables
        outputs:
            index of location in matrix
        '''
        Nk = self.Nk
        Nl = self.Nl
        SizeM = self.SizeM
        SizeMnoBC = self.SizeMnoBC
        Size_var = self.Size_var

        if (var not in self.model_variables):
            raise RuntimeError('variable not in model_variables')
        elif not (l >= 1 and l <= Nl):
            raise RuntimeError('l index out of bounds')
        elif not ((k >= 1 and k <= Nk) or ((k == 0 or k == Nk+1) and var in
                  self.boundary_variables)):
            raise RuntimeError('k index out of bounds')
        else:
            if ((var in self.boundary_variables) and (k == 0 or k == Nk+1)):
                if k == 0:
                    k_bound = 0
                elif k == Nk+1:
                    k_bound = 1
                return SizeMnoBC + Nl*2*self.boundary_variables.index(var) +\
                    k_bound*Nl + (l-1)
            else:
                return Size_var*self.model_variables.index(var) + (k-1) + (l-1)*Nk

    def get_variable(self, vector, var, returnBC=True):
        '''
        Takes a flat vector and a variable name, returns the variable in a
        np.matrix and the bottom and top boundary vectors
        inputs:
            vector: flat vector array with len == SizeM
            var: str of variable name in model
        outputs:
            if boundary exists: variable, bottom_boundary, top_boundary
            if no boundary exists: variable
        '''
        Nk = self.Nk
        Nl = self.Nl

        if (var not in self.model_variables):
            raise RuntimeError('variable not in model_variables')
        elif len(vector) != self.SizeM:
            raise RuntimeError('vector given is not correct length in this \
                               model')
        else:
            var_start = self.get_index(1, 1, var)
            var_end = self.get_index(Nk, Nl, var)+1
            variable = np.array(np.reshape(vector[var_start:var_end], (Nk, Nl), 'F'))
            if var in self.boundary_variables:
                bound_bot_start = self.get_index(0, 1, var)
                bound_bot_end = self.get_index(0, Nl, var)+1
                bound_top_start = self.get_index(Nk+1, 1, var)
                bound_top_end = self.get_index(Nk+1, Nl, var)+1
                bottom_boundary = np.array(np.reshape(vector[bound_bot_start:
                                             bound_bot_end], (1, Nl)))
                top_boundary = np.array(np.reshape(vector[bound_top_start:
                                                 bound_top_end], (1, Nl)))
                if returnBC:
                    return variable, bottom_boundary, top_boundary
                else:
                    return variable
            else:
                return variable

    def create_vector(self, variables, boundaries):
        '''
        Takes a set of variables and boundaries and creates a vector out of
        them.
        inputs:
            variables: list of (Nk x Nl) matrices or vectors for each model
                variable
            boundaries: list of 2 X Nl matrices or vectors for each boundary
        outputs:
            vector of size (SizeM x 1)
        '''
        Nk = self.Nk
        Nl = self.Nl
        vector = np.array([1])

        # Check Inputs:
        if len(variables) != len(self.model_variables):
            raise RuntimeError('Incorrect number of variable vectors passed')
        if len(boundaries) != len(self.boundary_variables):
            raise RuntimeError('Incorrect number of boundary vectors passed')
        for var in variables:
            vector = np.vstack((vector, np.reshape(var, (Nk*Nl, 1))))
        for bound in boundaries:
            vector = np.vstack((vector, np.reshape(bound, (2*Nl, 1))))
        return np.array(vector[1:])

    def add_gov_equation(self, name, variable):
        setattr(self, name, GovEquation(self, variable))

    def setup_SLEPc(self, nev=10, Target=None, Which='TARGET_MAGNITUDE'):
        self.EPS = SLEPc.EPS().create()
        self.EPS.setDimensions(10, PETSc.DECIDE)
        self.EPS.setOperators(self.A_SLEPc, self.M_SLEPc)
        self.EPS.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        self.EPS.setTarget(Target)
        self.EPS.setWhichEigenpairs(eval('self.EPS.Which.'+Which))
        self.EPS.setFromOptions()
        self.ST = self.EPS.getST()
        self.ST.setType(SLEPc.ST.Type.SINVERT)
        return self.EPS

    def solve_SLEPc(self, Target=None):
        self.EPS.solve()
        conv = self.EPS.getConverged()
        vs, ws = PETSc.Mat.getVecs(self.A_SLEPc)
        vals = []
        vecs = []
        for ind in range(conv):
            vals.append(self.EPS.getEigenpair(ind, ws))
            vecs.append(ws.getArray())
        return vals, vecs

    def save_mat_PETSc(self, filename, mat, type='Binary'):
        ''' Saves a Matrix in PETSc format '''
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename, 'w')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename, 'w')
        viewer(mat)

    def load_mat_PETSc(self, filename, type='Binary'):
        ''' Loads and returns a Matrix stored in PETSc format '''
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename, 'r')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename, 'r')
        return PETSc.Mat().load(viewer)

    def save_vec_PETSc(self, filename, vec, type='Binary'):
        ''' Saves a vector in PETSc format '''
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename, 'w')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename, 'w')
        viewer(vec)

    def load_vec_PETSc(self, filename, type='Binary'):
        ''' Loads and returns a vector stored in PETSc format '''
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename, 'r')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename, 'r')
        return PETSc.Mat().load(viewer)

    def save_model(self, filename):
        ''' Saves the model structure without the computed A and M matrices'''
        try:
            self.A
        except:
            pass
        else:
            A = self.A
            del self.A
        try:
            self.M
        except:
            pass
        else:
            M = self.M
            del self.M

        pkl.dump(self, open(filename, 'wb'))

        try:
            A
        except:
            pass
        else:
            self.A = A
        try:
            M
        except:
            pass
        else:
            self.M = M

    def make_D3sqMat(self):
        self.D3sq_rows = []
        self.D3sq_cols = []
        self.D3sq_vals = []
        for var in self.boundary_variables:
            self.add_gov_equation('D3sq_'+var, var)
            exec('self.D3sq_'+var+'.add_D3sq(\''+var+'\','+str(self.m)+')')
            exec('self.D3sq_rows = self.D3sq_'+var+'.rows')
            exec('self.D3sq_cols = self.D3sq_'+var+'.cols')
            exec('self.D3sq_vals = self.D3sq_'+var+'.vals')
        self.D3sqMat = coo_matrix((self.D3sq_vals, (self.D3sq_rows, self.D3sq_cols)),
                               shape=(self.SizeM, self.SizeM))
        return self.D3sqMat

    def make_dthMat(self):
        self.dth_rows = []
        self.dth_cols = []
        self.dth_vals = []
        for var in self.boundary_variables:
            self.add_gov_equation('dth_'+var, var)
            exec('self.dth_'+var+'.add_dth(\''+var+'\','+str(self.m)+')')
            exec('self.dth_rows += self.dth_'+var+'.rows')
            exec('self.dth_cols += self.dth_'+var+'.cols')
            exec('self.dth_vals += self.dth_'+var+'.vals')
        self.dthMat = coo_matrix((self.dth_vals, (self.dth_rows, self.dth_cols)),
                              shape=(self.SizeM, self.SizeM))
        return self.dthMat

    def make_dphMat(self):
        self.dph_rows = []
        self.dph_cols = []
        self.dph_vals = []
        for var in self.boundary_variables:
            self.add_gov_equation('dth_'+var, var)
            exec('self.dph_'+var+'.add_dth(\''+var+'\','+str(self.m)+')')
            exec('self.dph_rows += self.dth_'+var+'.rows')
            exec('self.dph_cols += self.dth_'+var+'.cols')
            exec('self.dph_vals += self.dth_'+var+'.vals')
        self.dthMat = coo_matrix((self.dph_vals, (self.dph_rows, self.dph_cols)),
                              shape=(self.SizeM, self.SizeM))
        return self.dphMat

    def make_Bobs(self):
        BrobsT = 2*np.ones((self.Nk+2, self.Nl+2))*cos(self.th)
        self.Brobs = BrobsT/self.B_star
        gradBrobsT = -2*np.ones((self.Nk+2, self.Nl+2))*sin(self.th)/self.R
        self.gradBrobs = gradBrobsT/self.B_star*self.r_star
        self.add_gov_equation('Bobs', 'ur')
        self.Bobs.add_term('uth', 'self.model.gradBrobs[k,l]')
        self.Bobs.add_Dth('uth', 'self.model.Brobs[k,l]')
        self.Bobs.add_Dph('uph', 'self.model.Brobs[k,l]')
        self.BobsMat = coo_matrix((self.Bobs.vals, (self.Bobs.rows, self.Bobs.cols)),
                                  shape=(self.SizeM, self.SizeM))
        return self.BobsMat

class GovEquation():
    def __init__(self, model, variable):
        self.rows = []
        self.cols = []
        self.vals = []
        self.variable = variable
        self.model = model

    def add_term(self, var, value, kdiff=0, ldiff=0, mdiff=0, k_vals=None,
                 l_vals=None):
        ''' Adds a term to the governing equation.
        By default, iterates over 1 < k < Nk and 1 < l < Nl
        with kdiff = ldiff = mdiff = 0
        inputs:
            str var:     model variable name to input for
            str value:   string expression to evaluate for value to input
            int kdiff:   offset to use for k
            int ldiff:   offset to use for l
            int mdiff:   offset to use for m
            list k_vals: list of int k values to iterate over
            list l_vals: list of int l values to iterate over
        output:
            none
        '''
        dr = self.model.dr
        r = self.model.r
        rp = self.model.rp
        rm = self.model.rm
        dth = self.model.dth
        th = self.model.th
        thm = self.model.thm
        thp = self.model.thp
        Nk = self.model.Nk
        Nl = self.model.Nl
        E = self.model.E
        Prm = self.model.Prm
        G = self.model.G
        Br = self.model.Br
        Bth = self.model.Bth
        U0 = self.model.U0
        m = self.model.m

        if l_vals is None:
            l_vals = range(1, Nl+1)
        if k_vals is None:
            k_vals = range(1, Nk+1)

        # Check Inputs:
        if var not in self.model.model_variables:
            raise RuntimeError('variable not in model')

        for l in l_vals:
            for k in k_vals:
                if l > Nl or l < 1:
                    raise RuntimeError('l out of bounds')
                if k > Nk+1 or k < 0:
                    raise RuntimeError('k out of bounds')
                if type(value) in (int, complex, float):
                    value_computed = value
                else:
                    try:
                        value_computed = eval(value, globals(), locals())
                    except:
                        import ipdb; ipdb.set_trace()
                        raise RuntimeError('Problem evaluating term')

                if not ((l+ldiff <= 0) or (l+ldiff >= Nl+1) or value_computed == 0.0):
                    self.rows.append(self.model.get_index(k, l, self.variable))
                    self.cols.append(self.model.get_index(k+kdiff, l+ldiff, var))
                    self.vals.append(value_computed)

    def add_bc(self, var, value, k, kdiff=0, k_vals=None, l_vals=None):
        dr = self.model.dr
        r = self.model.r
        rp = self.model.rp
        rm = self.model.rm
        dth = self.model.dth
        th = self.model.th
        thm = self.model.thm
        thp = self.model.thp
        Nk = self.model.Nk
        Nl = self.model.Nl
        E = self.model.E
        Prm = self.model.Prm
        G = self.model.G
        Br = self.model.Br
        Bth = self.model.Bth
        U0 = self.model.U0
        m = self.model.m

        if l_vals is None:
            l_vals = range(1, Nl+1)

        if var not in self.model.boundary_variables:
            raise RuntimeError('variable is not in boundary_variables')

        for l in l_vals:
            self.rows.append(self.model.get_index(k, l, var))
            self.cols.append(self.model.get_index(k+kdiff, l, var))
            self.vals.append(eval(value, globals(), locals()))

    def add_value(self, value, row, col):
        '''
        Adds a term to a specific index.
        value: term to add
        row: dictionary containing 'k','l',and 'var'
        col: dictionary containing 'k','l',and 'var'
        '''
        self.rows.append(self.model.get_index(row['k'], row['l'], row['var']))
        self.cols.append(self.model.get_index(col['k'], col['l'], col['var']))
        self.vals.append(eval(value, globals(), locals()))

    def add_Dr(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*+(rp[k]/r[k])**2.0 / (2.0*dr)',
                      kdiff=+1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*-(rm[k]/r[k])**2.0 / (2.0*dr)',
                      kdiff=-1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*-(sin(thp[l])/sin(th[l]))/(4.0*r[k])',
                      ldiff=+1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*-(sin(thm[l])/sin(th[l]))/(4.0*r[k])',
                      ldiff=-1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const) +
                      '*-(sin(thp[l])+sin(thm[l]))/(4.0*r[k]*sin(th[l]))',
                      k_vals=k_vals, l_vals=l_vals)

    def add_Dth(self, var, const=1, k_vals=None, l_vals=None):
        self.add_term(var, str(const) +
                      '*+(sin(thp[l])/sin(th[l]))/(2.0*r[k]*dth)', ldiff=+1,
                      k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const) +
                      '*-(sin(thm[l])/sin(th[l]))/(2.0*r[k]*dth)', ldiff=-1,
                      k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const) +
                      '*((sin(thp[l])-sin(thm[l]))/(2.0*dth) -\
                      cos(th[l]))/(r[k]*sin(th[l]))', k_vals=k_vals,
                      l_vals=l_vals)

    def add_Dph(self, var, const=1, k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*1j*m/(r[k]*sin(th[l]))')

    def add_dr(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*rp[k]**2/(2*r[k]**2*dr)', kdiff=+1,
                      k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*-(rm[k]**2)/(2*r[k]**2*dr)',
                      kdiff=-1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*1/r[k]', k_vals=k_vals,
                      l_vals=l_vals)

    def add_dth(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*sin(thp[l])/(2*sin(th[l])*r[k]*dth)',
                      ldiff=+1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*-sin(thm[l])/(2*sin(th[l])*r[k]*dth)',
                      ldiff=-1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const) +
                      '*(sin(thp[l])-sin(thm[l]))/(2*sin(th[l])*r[k]*dth)',
                      k_vals=k_vals, l_vals=l_vals)

    def add_dph(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*1j*m/(r[k]*sin(th[l]))',
                      k_vals=k_vals, l_vals=l_vals)

    def add_dthsq(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*-(sin(thp[l])+sin(thm[l]))/(sin(th[l])*r[k]**2*dth**2)', k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*sin(thp[l])/(sin(th[l])*r[k]**2*dth**2)', ldiff=+1, k_vals=k_vals,
                      l_vals=l_vals)
        self.add_term(var, str(const)+'*sin(thm[l])/(sin(th[l])*r[k]**2*dth**2)', ldiff=-1, k_vals=k_vals,
                      l_vals=l_vals)

    def add_D3sq(self, var, const=1., k_vals=None, l_vals=None):
        self.add_term(var, str(const)+'*(rp[k]/(r[k]*dr))**2', kdiff=+1,
                      k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*(rm[k]/(r[k]*dr))**2', kdiff=-1,
                      k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*sin(thp[l])/(sin(th[l])*r[k]**2*dth**2)',
                      ldiff=+1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const)+'*sin(thm[l])/(sin(th[l])*r[k]**2*dth**2)',
                      ldiff=-1, k_vals=k_vals, l_vals=l_vals)
        self.add_term(var, str(const) +
                      '*((-rp[k]**2-rm[k]**2)/(r[k]**2*dr**2) -\
                      (sin(thp[l])+sin(thm[l]))/(sin(th[l])*r[k]**2*dth**2) -\
                      m**2/(r[k]**2*sin(th[l])**2))', k_vals=k_vals,
                      l_vals=l_vals)

    def get_coo_matrix(self):
        return coo_matrix((self.vals, (self.rows, self.cols)),
                          shape=(self.model.SizeM, self.model.SizeM))

    def get_csr_matrix(self):
        return csr_matrix((self.vals, (self.rows, self.cols)),
                          shape=(self.model.SizeM, self.model.SizeM))

    def todense(self):
        return self.get_coo_matrix().todense()


class csr_matrix(scipy.sparse.csr.csr_matrix):
    ''' Subclass to allow conversion to PETSc matrix format'''
    def toPETSc(self, epsilon=1e-12):
        ind = self.nonzero()
        Mat = PETSc.Mat().createAIJ(size=self.shape)
        Mat.setUp()
        for i, j, val in zip(ind[0], ind[1], self.data):
            Mat.setValue(i, j, val)
        # If any diagnoal elements are zero, replace with epsilon
        for ind, val in enumerate(self.diagonal()):
            if val == 0.:
                Mat.setValue(ind, ind, epsilon)
        Mat.assemble()
        return Mat

    def tocoo(self, copy=True):
        '''Overridden method to allow converstion to PETsc matrix

        Original Documentation:
        Return a COOrdinate representation of this matrix

        When copy=False the index and data arrays are not copied.
        '''
        major_dim, minor_dim = self._swap(self.shape)
        data = self.data
        minor_indices = self.indices
        if copy:
            data = data.copy()
            minor_indices = minor_indices.copy()
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        scipy.sparse.compressed._sparsetools.expandptr(major_dim,
                                                       self.indptr,
                                                       major_indices)
        row, col = self._swap((major_indices, minor_indices))
        return coo_matrix((data, (row, col)), self.shape)


class coo_matrix(scipy.sparse.coo.coo_matrix):
    ''' Subclass to allow conversion to PETSc matrix format'''
    def toPETSc(self, epsilon=1e-12):
        csr_mat = self.tocsr()
        ind = csr_mat.nonzero()
        Mat = PETSc.Mat().createAIJ(size=self.shape)
        Mat.setUp()
        for i, j, val in zip(ind[0], ind[1], csr_mat.data):
            Mat.setValue(i, j, val)
        # If any diagnoal elements are zero, replace with epsilon
        for ind, val in enumerate(self.diagonal()):
            if val == 0.:
                Mat.setValue(ind, ind, epsilon)
        Mat.assemble()
        del csr_mat
        return Mat

    def toPETSc_unassembled(self, epsilon=1e-10):
        csr_mat = self.tocsr()
        ind = csr_mat.nonzero()
        Mat = PETSc.Mat().createAIJ(size=self.shape)
        Mat.setUp()
        for i, j, val in zip(ind[0], ind[1], csr_mat.data):
            Mat.setValue(i, j, val)
        # If any diagnoal elements are zero, replace with epsilon
        for ind, val in enumerate(self.diagonal()):
            if val == 0.:
                Mat.setValue(ind, ind, epsilon)
        del csr_mat
        return Mat

    def tocsr(self):
        '''Overridden method to return csr matrix with toPETsc function

        Original Documentation:
        Return a copy of this matrix in Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        '''
        from scipy.sparse.sputils import get_index_dtype
        
        if self.nnz == 0:
            return csr_matrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = get_index_dtype((self.row, self.col),
                                                         maxval=max(self.nnz,
                                                                    N))
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty(self.nnz, dtype=idx_dtype)
            data = np.empty(self.nnz,
                            dtype=scipy.sparse.coo.upcast(self.dtype))

            scipy.sparse.coo.coo_tocsr(M, N, self.nnz,
                                       self.row.astype(idx_dtype),
                                       self.col.astype(idx_dtype),
                                       self.data,
                                       indptr,
                                       indices,
                                       data)

            A = csr_matrix((data, indices, indptr), shape=self.shape)
            A.sum_duplicates()
            return A
