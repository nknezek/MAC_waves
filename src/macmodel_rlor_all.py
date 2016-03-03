import macmodel
import numpy as np

class Model(macmodel.Model):
    def make_A(self):
        self.make_A_noCCBC(self)
        self.add_CCBC(self)
        return self.A

    def make_A_noCCBC(self):
        Nk = self.Nk
        Nl = self.Nl

        '''
        Creates the A matrix (M*l*x = A*x)
        m: azimuthal fourier mode to compute
        '''

        ################################
        # Momentum Equation ############
        ################################
        # R-momentum
        self.add_gov_equation('rmom', 'ur')
        self.rmom.add_Dr('p', const=-1)
        self.rmom.add_term('r_disp', '-G[k, l]')
        # self.rmom.add_term('uph', '2.0*sin(th[l])')
        self.rmom.add_D3sq('ur', 'E')
        self.rmom.add_dth('uth', '-E/r[k]')
        self.rmom.add_dph('uph', '-E/r[k]')
        # self.rmom.add_dr('br', 'E/Prm*Br[k,l]')
        # self.rmom.add_dr('bth', '-E/Prm*Bth[k,l]')
        # self.rmom.add_dr('bph', '-E/Prm*Bph[k,l]')
        # self.rmom.add_dth('br', 'E/Prm*Bth[k,l]')
        # self.rmom.add_dth('bth', 'E/Prm*Br[k,l]')
        # self.rmom.add_dph('bph', 'E/Prm*Br[k,l]')
        # self.rmom.add_dph('br', 'E/Prm*Bph[k,l]')
        self.A_rows = self.rmom.rows
        self.A_cols = self.rmom.cols
        self.A_vals = self.rmom.vals
        del self.rmom

        # Theta-Momentum
        self.add_gov_equation('tmom', 'uth')
        self.tmom.add_Dth('p', -1)
        self.tmom.add_term('uph', '+2.0*cos(th[l])')
        self.tmom.add_D3sq('uth', 'E')
        self.tmom.add_dth('ur', 'E/r[k]')
        self.tmom.add_dph('uph', '-E*cos(th[l])/(sin(th[l])*r[k])')
        self.tmom.add_dr('bth', 'E/Prm*Br[k,l]')
        self.tmom.add_dr('br', '-E/Prm*Bth[k,l]')
        self.tmom.add_dth('bth', 'E/Prm*Bth[k,l]')
        self.tmom.add_dth('br', '-E/Prm*Br[k,l]')
        self.tmom.add_dth('bph', '-E/Prm*Bph[k,l]')
        self.tmom.add_dph('bph', 'E/Prm*Bth[k,l]')
        self.tmom.add_dph('bth', 'E/Prm*Bph[k,l]')
        self.A_rows += self.tmom.rows
        self.A_cols += self.tmom.cols
        self.A_vals += self.tmom.vals
        del self.tmom

        # Phi-Momentum
        self.add_gov_equation('pmom', 'uph')
        self.pmom.add_Dph('p', -1)
        self.pmom.add_term('uth', '-2.0*cos(th[l])')
        self.pmom.add_term('ur', '-2.0*sin(th[l])')
        self.pmom.add_D3sq('uph', 'E')
        self.pmom.add_dph('ur', 'E/r[k]')
        self.pmom.add_dph('uth', 'E*cos(th[l])/(sin(th[l])*r[k])')
        self.pmom.add_dr('bph', 'E/Prm*Br[k,l]')
        self.pmom.add_dr('br', '-E/Prm*Bph[k,l]')
        self.pmom.add_dth('bph', 'E/Prm*Bth[k,l]')
        self.pmom.add_dth('bth', 'E/Prm*Bph[k,l]')
        self.pmom.add_dph('bph', 'E/Prm*Bph[k,l]')
        self.pmom.add_dph('br', '-E/Prm*Br[k,l]')
        self.pmom.add_dph('bth', '-E/Prm*Bth[k,l]')
        self.A_rows += self.pmom.rows
        self.A_cols += self.pmom.cols
        self.A_vals += self.pmom.vals
        del self.pmom

        ################################
        # Lorentz Equation ##########
        ################################
        # r-Lorentz
        self.add_gov_equation('rlorentz', 'br')
        self.rlorentz.add_dth('ur', 'Bth[k,l]')
        self.rlorentz.add_dth('uth', '-Br[k,l]')
        self.rlorentz.add_dph('ur', 'Bph[k,l]')
        self.rlorentz.add_dph('uph', '-Br[k,l]')
        self.rlorentz.add_D3sq('br', 'E/Prm')
        self.rlorentz.add_dth('bth', '-E/Prm/r[k]')
        self.rlorentz.add_dph('bph', '-E/Prm/r[k]')
        self.A_rows += self.rlorentz.rows
        self.A_cols += self.rlorentz.cols
        self.A_vals += self.rlorentz.vals
        del self.rlorentz

#         B-divergence replaces r-lorentz
#        self.add_gov_equation('bdiv', 'br')
#        self.bdiv.add_dr('br')
#        self.bdiv.add_dth('bth')
#        self.bdiv.add_dph('bph')
#        self.A_rows += self.bdiv.rows
#        self.A_cols += self.bdiv.cols
#        self.A_vals += self.bdiv.vals
#        del self.bdiv

        # theta-Lorentz
        self.add_gov_equation('thlorentz', 'bth')
        self.thlorentz.add_dr('uth', 'Br[k,l]')
        self.thlorentz.add_dr('ur', '-Bth[k,l]')
        self.thlorentz.add_dph('uth', 'Bph[k, l]')
        self.thlorentz.add_dph('uph', '-Bth[k, l]')
        self.thlorentz.add_D3sq('bth', 'E/Prm')
        self.thlorentz.add_dth('br', 'E/Prm/r[k]')
        self.thlorentz.add_dph('bph', '-E/Prm*cos(th[l])/(sin(th[l])*r[k])')
        self.A_rows += self.thlorentz.rows
        self.A_cols += self.thlorentz.cols
        self.A_vals += self.thlorentz.vals
        del self.thlorentz

        # phi-Lorentz
        self.add_gov_equation('phlorentz', 'bph')
        self.phlorentz.add_dr('uph', 'Br[k,l]')
        self.phlorentz.add_dr('ur', '-Bph[k,l]')
        self.phlorentz.add_dth('uph', 'Bth[k,l]')
        self.phlorentz.add_dth('uth', '-Bph[k,l]')
        self.phlorentz.add_D3sq('bph', 'E/Prm')
        self.phlorentz.add_dph('br', 'E/Prm/r[k]')
        self.phlorentz.add_dph('bth', 'E/Prm*cos(th[l])/(sin(th[l])*r[k])')
        self.A_rows += self.phlorentz.rows
        self.A_cols += self.phlorentz.cols
        self.A_vals += self.phlorentz.vals
        del self.phlorentz

        # Divergence (Mass Conservation) #########
        self.add_gov_equation('div', 'p')
        self.div.add_dr('ur')
        self.div.add_dth('uth')
        self.div.add_dph('uph')
        self.A_rows += self.div.rows
        self.A_cols += self.div.cols
        self.A_vals += self.div.vals
        del self.div

        # Displacement Equation #########
        self.add_gov_equation('rdisp', 'r_disp')
        self.rdisp.add_term('ur', '1')
        self.A_rows += self.rdisp.rows
        self.A_cols += self.rdisp.cols
        self.A_vals += self.rdisp.vals
        del self.rdisp

        # Boundary Conditions
        self.add_gov_equation('BC', 'p')

        self.BC.add_bc('ur', 'r[k]**2', 0)
        self.BC.add_bc('ur', 'r[k]**2', 0, kdiff=1)
        self.BC.add_bc('ur', 'r[k]**2', Nk+1)
        self.BC.add_bc('ur', 'r[k]**2', Nk+1, kdiff=-1)

        self.BC.add_bc('uth', 'r[k]**2', 0)
        self.BC.add_bc('uth', '-r[k]**2', 0, kdiff=1)
        self.BC.add_bc('uth', 'r[k]**2', Nk+1)
        self.BC.add_bc('uth', '-r[k]**2', Nk+1, kdiff=-1)

        self.BC.add_bc('uph', 'r[k]**2', 0)
        self.BC.add_bc('uph', '-r[k]**2', 0, kdiff=1)
        self.BC.add_bc('uph', 'r[k]**2', Nk+1)
        self.BC.add_bc('uph', '-r[k]**2', Nk+1, kdiff=-1)

        self.BC.add_bc('p', '1.', 0)
        self.BC.add_bc('p', '-1.', 0, kdiff=1)
        self.BC.add_bc('p', '1.', Nk+1)
        self.BC.add_bc('p', '-1.', Nk+1, kdiff=-1)

        self.BC.add_bc('br', 'r[k]**2', 0)
        self.BC.add_bc('br', '-r[k]**2', 0, kdiff=1)
        self.BC.add_bc('br', 'r[k]**2', Nk+1)
        self.BC.add_bc('br', '-r[k]**2', Nk+1, kdiff=-1)

        self.BC.add_bc('bth', 'r[k]**2', Nk+1)
        self.BC.add_bc('bth', 'r[k]**2', Nk+1, kdiff=-1)

        self.BC.add_bc('bph', 'r[k]**2', Nk+1)
        self.BC.add_bc('bph', 'r[k]**2', Nk+1, kdiff=-1)

        self.A_rows += self.BC.rows
        self.A_cols += self.BC.cols
        self.A_vals += self.BC.vals
        del self.BC
        self.A_noCCBC = macmodel.coo_matrix((self.A_vals, (self.A_rows, self.A_cols)),
                                   shape=(self.SizeM, self.SizeM))
        del self.A_vals, self.A_rows, self.A_cols
        return self.A_noCCBC

    def add_CCBC(self):
        # Conducting Core at CFB ####
        Nk = self.Nk
        Nl = self.Nl
        E = self.E
        Prm = self.Prm
        delta_C = self.delta_C
        r_star = self.r_star
        Br = self.Br
        dr = self.dr

        self.add_gov_equation('CCBC', 'p')
        for l in range(1, Nl+1):
            row = {'k': 0, 'l': l, 'var': 'bth'}
            self.CCBC.add_value(str(Br[0, l]/2.), row,
                              {'k': 0, 'l': l, 'var': 'uth'})
            self.CCBC.add_value(str(Br[1, l]/2.), row,
                              {'k': 1, 'l': l, 'var': 'uth'})
            self.CCBC.add_value(str(E/Prm*(-1/dr - r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 0, 'l': l, 'var': 'bth'})
            self.CCBC.add_value(str(E/Prm*(1/dr - r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 1, 'l': l, 'var': 'bth'})
        for l in range(1, Nl+1):
            row = {'k': 0, 'l': l, 'var': 'bph'}
            self.CCBC.add_value(str(Br[0, l]/2.), row,
                              {'k': 0, 'l': l, 'var': 'uph'})
            self.CCBC.add_value(str(Br[1, l]/2.), row,
                              {'k': 1, 'l': l, 'var': 'uph'})
            self.CCBC.add_value(str(E/Prm*(-1/dr - r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 0, 'l': l, 'var': 'bph'})
            self.CCBC.add_value(str(E/Prm*(1/dr - r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 1, 'l': l, 'var': 'bph'})
        data = np.concatenate((self.A_noCCBC.data, self.CCBC.vals))
        rows = np.concatenate((self.A_noCCBC.row, self.CCBC.rows))
        cols = np.concatenate((self.A_noCCBC.col, self.CCBC.cols))
        self.A = macmodel.coo_matrix((data, (rows, cols)),
                                   shape=(self.SizeM, self.SizeM))
        del self.CCBC
        return self.A

    def make_B(self):
        '''
        Creates the B matrix (B*l*x = A*x)
        m: azimuthal fourier mode to compute
        '''
#        self.add_gov_equation('B_uth', 'uth')
#        self.B_uth.add_term('uth', '1')
#        self.B_rows = self.B_uth.rows
#        self.B_cols = self.B_uth.cols
#        self.B_vals = self.B_uth.vals
#        del self.B_uth

#        self.add_gov_equation('B_uph', 'uph')
#        self.B_uph.add_term('uph', '1')
#        self.B_rows += self.B_uph.rows
#        self.B_cols += self.B_uph.cols
#        self.B_vals += self.B_uph.vals
#        del self.B_uph
#
        self.add_gov_equation('B_rlorentz', 'br')
        self.B_rlorentz.add_term('br', '1')
        self.B_rows = self.B_rlorentz.rows
        self.B_cols = self.B_rlorentz.cols
        self.B_vals = self.B_rlorentz.vals
        del self.B_rlorentz

        self.add_gov_equation('B_thlorentz', 'bth')
        self.B_thlorentz.add_term('bth', '1')
        self.B_rows += self.B_thlorentz.rows
        self.B_cols += self.B_thlorentz.cols
        self.B_vals += self.B_thlorentz.vals
        del self.B_thlorentz

        self.add_gov_equation('B_phlorentz', 'bph')
        self.B_phlorentz.add_term('bph', '1')
        self.B_rows += self.B_phlorentz.rows
        self.B_cols += self.B_phlorentz.cols
        self.B_vals += self.B_phlorentz.vals
        del self.B_phlorentz

        self.add_gov_equation('B_rdisp', 'r_disp')
        self.B_rdisp.add_term('r_disp', '1')
        self.B_rows += self.B_rdisp.rows
        self.B_cols += self.B_rdisp.cols
        self.B_vals += self.B_rdisp.vals
        del self.B_rdisp
        self.B = macmodel.coo_matrix((self.B_vals, (self.B_rows, self.B_cols)),
                                   shape=(self.SizeM, self.SizeM))
        del self.B_vals, self.B_rows, self.B_cols
        return self.B

