import macmodel
import numpy as np
from numpy import sin, cos, tan

class Model(macmodel.Model):

    def make_A(self):
        Nk = self.Nk
        Nl = self.Nl
        E = self.E
        Pm = self.Pm
        N = self.N
        th = self.th
        Br = self.Br
        Bth = self.Bth
        Bph = self.Bph
        '''
        Creates the A matrix (M*l*x = A*x)
        m: azimuthal fourier mode to compute
        '''

        ################################
        # Momentum Equation ############
        ################################
        self.A_rows = []
        self.A_cols = []
        self.A_vals = []

        # Theta-Momentum
        self.add_gov_equation('tmom', 'uth')
        self.tmom.add_dthP('p', C= -1)
        self.tmom.add_term('uph', 2.0*cos(th))
        self.tmom.add_d2_bd0('uth', C= E)
        self.tmom.add_d2th_ph('uph', C= E)
        self.A_rows += self.tmom.rows
        self.A_cols += self.tmom.cols
        self.A_vals += self.tmom.vals
        del self.tmom

        # Phi-Momentum
        self.add_gov_equation('pmom', 'uph')
        self.pmom.add_dphP('p', C= -1)
        self.pmom.add_term('uth', -2.0*cos(th))
        self.pmom.add_d2_bd0('uph', C= E)
        self.pmom.add_d2ph_th('uth', C= E)
        self.A_rows += self.pmom.rows
        self.A_cols += self.pmom.cols
        self.A_vals += self.pmom.vals
        del self.pmom

        # Divergence (Mass Conservation) #########
        self.add_gov_equation('div', 'p')
        self.div.add_dth('uth')
        self.div.add_dph('uph')
        self.A_rows += self.div.rows
        self.A_cols += self.div.cols
        self.A_vals += self.div.vals
        del self.div

        self.A = macmodel.coo_matrix((self.A_vals, (self.A_rows, self.A_cols)),
                                   shape=(self.SizeM, self.SizeM))
        del self.A_vals, self.A_rows, self.A_cols
        return self.A

    def make_B(self):
        '''
        Creates the B matrix (B*l*x = A*x)
        m: azimuthal fourier mode to compute
        '''
        ones = np.ones((self.Nk+2, self.Nl))
        self.B_rows = []
        self.B_cols = []
        self.B_vals = []

        self.add_gov_equation('B_uth', 'uth')
        self.B_uth.add_term('uth', ones)
        self.B_rows = self.B_uth.rows
        self.B_cols = self.B_uth.cols
        self.B_vals = self.B_uth.vals
        del self.B_uth

        self.add_gov_equation('B_uph', 'uph')
        self.B_uph.add_term('uph', ones)
        self.B_rows += self.B_uph.rows
        self.B_cols += self.B_uph.cols
        self.B_vals += self.B_uph.vals
        del self.B_uph

        self.B = macmodel.coo_matrix((self.B_vals, (self.B_rows, self.B_cols)),
                                   shape=(self.SizeM, self.SizeM))
        del self.B_vals, self.B_rows, self.B_cols
        return self.B