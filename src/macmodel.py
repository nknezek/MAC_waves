import numpy as np
import scipy
import scipy.sparse as sparse
from numpy import sin
from numpy import cos
import sys
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
opts = PETSc.Options()
import cPickle as pkl
import copy

class Model():
    def __init__(self, model_variables, boundary_variables, 
                 model_parameters, physical_constants):
        self.model_variables = model_variables
        self.boundary_variables = boundary_variables

        for key in model_parameters:
            exec('self.'+str(key)+' = model_parameters[\''+str(key)+'\']')
        for key in physical_constants:
            exec('self.'+str(key)+' = physical_constants[\''+str(key)+'\']')

        self.calculate_nondimensional_parameters()
        self.set_up_grid(self.R, self.h)

    def set_up_grid(self, R, h):
        """
        Creates the r and theta coordinate vectors
        inputs:
            R: radius of outer core in m
            h: layer thickness in m
        outputs: None
        """
        self.R = R
        self.h = h
        self.Size_var = self.Nk*self.Nl
        self.SizeMnoBC = len(self.model_variables)*self.Size_var
        self.SizeM = self.SizeMnoBC + 2*len(self.boundary_variables)*self.Nl
        self.rmin = (self.R-self.h)/self.r_star
        self.rmax = self.R/self.r_star
        self.dr = (self.rmax-self.rmin)/(self.Nk)
        self.r = np.linspace(self.rmin-self.dr/2.,self.rmax+self.dr/2.,num=self.Nk+2.) # r value at center of each cell
        self.rm = np.linspace(self.rmin-self.dr,self.rmax,num=self.Nk+2.) # r value at plus border (top) of cell
        self.rp = np.linspace(self.rmin,self.rmax+self.dr,num=self.Nk+2.) # r value at minus border (bottom) of cell
        self.dth = np.pi/(self.Nl)
        self.th = np.linspace(-self.dth/2.,np.pi+self.dth/2.,num=self.Nl+2.) # theta value at center of cell
        self.thm = np.linspace(-self.dth,np.pi,num=self.Nl+2.) # theta value at plus border (top) of cell
        self.thp = np.linspace(0,np.pi+self.dth,num=self.Nl+2.)
        return None
    
    def calculate_nondimensional_parameters(self):
        """
        Calculates the non-dimensional parameters in model from the physical constants.
        """
        self.t_star = 1/self.Omega  # seconds
        self.r_star = self.R  # meters
        self.P_star = self.rho* self.r_star**2/self.t_star**2
        self.B_star = (self.eta*self.mu_0*self.rho/self.t_star)**0.5
        self.u_star = self.r_star/self.t_star
        self.E = self.nu*self.t_star/self.r_star**2
        self.Prm = self.nu/self.eta
        return None
    
    def set_Br(self,Br):
        """ Sets the backgound magnetic field in Tesla"""
        self.Br = Br
        self.B0 = self.Br/self.B_star
        return None
    
    def set_dipole_Br(self,Bd):
        self.Bd = Bd
        self.Br = np.ones((self.Nk+2,self.Nl+2))*cos(self.th)*Bd
        self.B0 = self.Br/self.B_star
        return None
        
    def set_Uphi(self,Uphi):
        """Sets the background velocity field in m/s"""
        self.Uphi = Uphi
        self.U0 = self.Uphi*self.r_star/self.t_star
        return None
    
    def set_buoyancy(self,drho_dr):
        """Sets the buoyancy structure of the layer"""
        self.omega_g = np.sqrt(-self.g/self.rho*drho_dr)
        self.G = self.omega_g**2*self.t_star**2
        
    def get_index(self, k, l, *args):
        """
        Takes coordinates for a point, gives back index in matrix.
        inputs:
            k: k grid value from 1 to K (or 0 to K+1 for variables with ghost points)
            l: l grid value from 1 to L
            var: 'ur', 'uth', 'uph', or 'p'
        outputs:
            index of location in matrix
        """
        Nk = self.Nk
        Nl = self.Nl
        SizeM = self.SizeM
        SizeMnoBC = self.SizeMnoBC
        Size_var = self.Size_var

        if len(args) > 2 or len(args) < 1:
            raise RuntimeError('incorrect number of arguments passed')
        else:
            if len(args) == 1:
                m = self.m_values[0]
                var = args[0]
            elif len(args) == 2:
                m = args[0]
                var = args[1]

        if (var not in self.model_variables):
            raise RuntimeError('variable not in model_variables')
        elif m not in self.m_values:
            raise RuntimeError('m not in modes to simulate')
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
                return self.m_values.index(m)*SizeM + SizeMnoBC +\
                    Nl*2*self.boundary_variables.index(var) + k_bound*Nl +\
                    (l-1)
            else:
                return self.m_values.index(m)*SizeM + \
                    Size_var*self.model_variables.index(var) + (k-1) + (l-1)*Nk

    def get_variable(self, vector, var, *args):
        """
        Takes a flat vector and a variable name, returns the variable in a
        np.matrix and the bottom and top boundary vectors
        inputs:
            vector: flat vector array with len == SizeM
            var: str of variable name in model
        outputs:
            if boundary exists: variable, bottom_boundary, top_boundary
            if no boundary exists: variable
        """
        Nk = self.Nk
        Nl = self.Nl
        if len(args) > 1 or len(args) < 0:
            raise RuntimeError('incorrect number of arguments passed')
        else:
            if len(args) == 0:
                m = self.m_values[0]
            elif len(args) == 1:
                m = args[0]
        if (var not in self.model_variables):
            raise RuntimeError('variable not in model_variables')
        elif len(vector) != self.SizeM:
            raise RuntimeError('vector given is not correct length in this \
                               model')

        else:
            var_start = self.get_index(1, 1, m, var)
            var_end = self.get_index(Nk, Nl, m, var)+1
            variable = np.reshape(vector[var_start:var_end], (Nk, Nl), 'F')
            if var in self.boundary_variables:
                bound_bot_start = self.get_index(0, 1, m, var)
                bound_bot_end = self.get_index(0, Nl, m, var)+1
                bound_top_start = self.get_index(Nk+1, 1, m, var)
                bound_top_end = self.get_index(Nk+1, Nl, m, var)+1
                bottom_boundary = np.reshape(vector[bound_bot_start:
                                             bound_bot_end], (1, Nl))
                top_boundary = np.reshape(vector[bound_top_start:
                                                 bound_top_end], (1, Nl))
                return variable, bottom_boundary, top_boundary
            else:
                return variable

    def create_vector(self, variables, boundaries):
        """
        Takes a set of variables and boundaries and creates a vector out of
        them.
        inputs:
            variables: list of (Nk x Nl) matrices or vectors for each model
                variable
            boundaries: list of 2 X Nl matrices or vectors for each boundary
        outputs:
            vector of size (SizeM x 1)
        """
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

#    def convert_A_SLEPc(self):
#        """ Converts the scipy sparse A matrix to PETSc format"""
#        try:
#            self.A
#        except NameError:
#            self.A = self.make_A()
#
#        # Set up A matrix
#        self.A_SLEPc = PETSc.Mat().createAIJ([self.SizeM,  self.SizeM])
#        self.A_SLEPc.setType(PETSc.Mat.Type.AIJ)
#        self.A_SLEPc.setUp()
#        for key,  val in self.A.todok().iteritems():
#            self.A_SLEPc[key[0],  key[1]] = val
#        self.A_SLEPc.assemble()
#        return self.A_SLEPc
#
#    def convert_M_SLEPc(self, epsilon=1e-7):
#        """ Converts the scipy sparse M matrix to PETSc format"""
#        try:
#            self.M
#        except NameError:
#            self.M = self.make_M()
#        # Set up M matrix
#        self.M_SLEPc = PETSc.Mat().createAIJ([self.SizeM,  self.SizeM])
#        self.M_SLEPc.setType(PETSc.Mat.Type.AIJ)
#        self.M_SLEPc.setUp()
#        for key,  val in self.M.todok().iteritems():
#            self.M_SLEPc[key[0],  key[1]] = val
#        # If any diagnoal elements are zero, replace with epsilon
#        for ind in range(self.SizeM):
#            if self.M[ind, ind] == 0.:
#                self.M_SLEPc[ind, ind] = epsilon
#        self.M_SLEPc.assemble()
#        return self.M_SLEPc
        
    def save_mat_PETSc(self, filename, mat, type='Binary'):
        """ Saves a Matrix in PETSc format """
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename,'w')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename,'w')
        viewer(mat)

    def load_mat_PETSc(self, filename, type='Binary'):
        """ Loads and returns a Matrix stored in PETSc format """
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename,'r')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename,'r')
        return PETSc.Mat().load(viewer)

    def save_vec_PETSc(self, filename, vec, type='Binary'):
        """ Saves a vector in PETSc format """
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename,'w')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename,'w')
        viewer(vec)

    def load_vec_PETSc(self, filename, type='Binary'):
        """ Loads and returns a vector stored in PETSc format """
        if type == 'Binary':
            viewer = PETSc.Viewer().createBinary(filename,'r')
        elif type == 'ASCII':
            viewer = PETSc.Viewer().createASCII(filename,'r')
        return PETSc.Mat().load(viewer)
        
    def save_model(self, filename):
        """ Saves the model structure without the computed A and M matrices"""
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

        pkl.dump(self, open(filename,'wb'))

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

class GovEquation():
    def __init__(self, model, variable):
        self.rows = []
        self.cols = []
        self.vals = []
        self.variable = variable
        self.model = model

    def add_term(self, var, value, m, kdiff=0, ldiff=0, mdiff=0, first_l=1,
                last_l=None, first_k=1, last_k=None):
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

        B0 = self.model.B0
        U0 = self.model.U0

        if last_l is None:
            last_l = Nl
        if last_k is None:
            last_k = Nk

        # Check Inputs:
        if first_l > Nl or first_l < 1:
            raise RuntimeError('first_l out of bounds')
        if last_l > Nl or last_l < 1:
            raise RuntimeError('last_l out of bounds')
        if first_k > Nk+1 or first_k < 0:
            raise RuntimeError('first_k out of bounds')
        if last_k > Nk+1 or last_k < 0:
            raise RuntimeError('last_k out of bounds')
        if var not in self.model.model_variables:
            raise RuntimeError('variable not in model')

        for l in range(first_l, last_l+1):
            for k in range(first_k, last_k+1):
                try:
                    value_computed = eval(value, globals(), locals())
                except:
                    raise RuntimeError('Problem evaluating term')
                if not np.isclose(value_computed, 0.0, atol=Nl*1E-12,
                                  rtol=Nl*1E-12):
                    self.rows.append(self.model.get_index(k, l, self.variable))
                    self.cols.append(self.model.get_index(k+kdiff, l+ldiff,
                                     var))
                    self.vals.append(value_computed)

    def add_bc(self, var, value, k, m, kdiff=0, first_l=1, last_l=None):
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

        B0 = self.model.B0
        U0 = self.model.U0

        if last_l is None:
            last_l = Nl

        if var not in self.model.boundary_variables:
            raise RuntimeError('variable is not in boundary_variables')

        for l in range(first_l, last_l+1):
            self.rows.append(self.model.get_index(k, l, var))
            self.cols.append(self.model.get_index(k+kdiff, l, var))
            self.vals.append(eval(value, globals(), locals()))

    def add_value(self, value, row, col):
        """
        Adds a term to a specific index.
        value: term to add
        row: dictionary containing 'k','l',and 'var'
        col: dictionary containing 'k','l',and 'var'
        """
        self.rows.append(self.model.get_index(row['k'], row['l'], row['var']))
        self.cols.append(self.model.get_index(col['k'], col['l'], col['var']))
        self.vals.append(eval(value, globals(), locals()))

    def add_Dr(self, var, m, const=1.):
        self.add_term(var, str(const)+'*+(rp[k]/r[k])**2.0 / (2.0*dr)', m,
                      kdiff=+1)
        self.add_term(var, str(const)+'*-(rm[k]/r[k])**2.0 / (2.0*dr)', m,
                      kdiff=-1)
        self.add_term(var, str(const)+'*-(sin(thp[l])/sin(th[l]))/(4.0*r[k])',
                      m, ldiff=+1)
        self.add_term(var, str(const)+'*-(sin(thm[l])/sin(th[l]))/(4.0*r[k])',
                      m, ldiff=-1)
        self.add_term(var, str(const) +
                      '*-(sin(thp[l])+sin(thm[l]))/(4.0*r[k]*sin(th[l]))', m)

    def add_Dth(self, var, m, const=1):
        self.add_term(var, str(const) +
                     '*+(sin(thp[l])/sin(th[l]))/(2.0*r[k]*dth)', m, ldiff=+1)
        self.add_term(var, str(const) +
                     '*-(sin(thm[l])/sin(th[l]))/(2.0*r[k]*dth)', m, ldiff=-1)
        self.add_term(var, str(const) +
                     '*((sin(thp[l])-sin(thm[l]))/(2.0*dth) -\
                     cos(th[l]))/(r[k]*sin(th[l]))', m)

    def add_Dph(self, var, m, const=1):
        self.add_term(var, str(const)+'*1j*m/(r[k]*sin(th[l]))', m)

    def add_dr(self, var, m, const=1.):
        self.add_term(var, str(const)+'*rp[k]**2/(2*r[k]**2*dr)', m, kdiff=+1)
        self.add_term(var, str(const)+'*-(rm[k]**2)/(2*r[k]**2*dr)', m,
                     kdiff=-1)
        self.add_term(var, str(const)+'*1/r[k]', m)

    def add_dth(self, var, m, const=1.):
        self.add_term(var, str(const)+'*sin(thp[l])/(2*sin(th[l])*r[k]*dth)',
                     m, ldiff=+1)
        self.add_term(var, str(const)+'*-sin(thm[l])/(2*sin(th[l])*r[k]*dth)',
                     m, ldiff=-1)
        self.add_term(var, str(const) +
                     '*(sin(thp[l])-sin(thm[l]))/(2*sin(th[l])*r[k]*dth)', m)

    def add_dph(self, var, m, const=1.):
        self.add_term(var, str(const)+'*1j*m/(r[k]*sin(th[l]))', m)

    def add_D3sq(self, var, m, const=1.):
        self.add_term(var, str(const)+'*(rp[k]/(r[k]*dr))**2', m, kdiff=+1)
        self.add_term(var, str(const)+'*(rm[k]/(r[k]*dr))**2', m, kdiff=-1)
        self.add_term(var, str(const) +
                     '*sin(thp[l])/(sin(th[l])*r[k]**2*dth**2)', m, ldiff=+1)
        self.add_term(var, str(const) +
                     '*sin(thm[l])/(sin(th[l])*r[k]**2*dth**2)', m, ldiff=-1)
        self.add_term(var, str(const) +
                     '*((-rp[k]**2-rm[k]**2)/(r[k]**2*dr**2) -\
                     (sin(thp[l])+sin(thm[l]))/(sin(th[l])*r[k]**2*dth**2) -\
                     m**2/(r[k]**2*sin(th[l])**2))', m)

    def get_coo_matrix(self):
        return coo_matrix((self.vals, (self.rows, self.cols)),
                          shape=(self.model.SizeM, self.model.SizeM))

    def get_csr_matrix(self):
        return csr_matrix((self.vals, (self.rows, self.cols)),
                          shape=(self.model.SizeM, self.model.SizeM))

    def todense(self):
        return self.get_coo_matrix().todense()


class csr_matrix(scipy.sparse.csr.csr_matrix):
    """ Subclass to allow conversion to PETSc matrix format"""
    def toPETSc(self, epsilon=1e-7):
        # Set up A matrix
        Mat = PETSc.Mat().createAIJ([self.shape[0],  self.shape[1]])
        Mat.setType(PETSc.Mat.Type.AIJ)
        Mat.setUp()
        for key,  val in self.todok().iteritems():
            Mat[key[0],  key[1]] = val
        # If any diagnoal elements are zero, replace with epsilon
        for ind in range(self.shape[0]):
            if self[ind, ind] == 0.:
                Mat[ind, ind] = epsilon
        Mat.assemble()
        return Mat

    def tocoo(self, copy=True):
        """Overridden method to allow converstion to PETsc matrix

        Original Documentation:        
        Return a COOrdinate representation of this matrix

        When copy=False the index and data arrays are not copied.
        """
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
    """ Subclass to allow conversion to PETSc matrix format"""
    def toPETSc(self, epsilon=1e-7):
        # Set up A matrix
        Mat = PETSc.Mat().createAIJ([self.shape[0],  self.shape[1]])
        Mat.setType(PETSc.Mat.Type.AIJ)
        Mat.setUp()
        for key,  val in self.todok().iteritems():
            Mat[key[0],  key[1]] = val
        # If any diagnoal elements are zero, replace with epsilon
        for ind in range(self.shape[0]):
            if self[ind, ind] == 0.:
                Mat[ind, ind] = epsilon
        Mat.assemble()
        return Mat

    def tocsr(self):
        """Overridden method to return csr matrix with toPETsc function
        
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

        """
        if self.nnz == 0:
            return csr_matrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = scipy.sparse.coo.get_index_dtype((self.row, self.col),
                                        maxval=max(self.nnz, N))
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty(self.nnz, dtype=idx_dtype)
            data = np.empty(self.nnz, dtype=scipy.sparse.coo.upcast(self.dtype))

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
