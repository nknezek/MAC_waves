import macmodel


class Model(macmodel.Model):
    def make_A(self, m):
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
        self.rmom.add_Dr('p', m, const=-1)
        self.rmom.add_term('r_disp', '-G[k, l]', m)
        self.rmom.add_D3sq('ur', m, 'E')
        self.rmom.add_dth('uth', m, '-E/r[k]')
        self.A = self.rmom.get_coo_matrix()
        del self.rmom

        # Theta-Momentum
        self.add_gov_equation('tmom', 'uth')
        self.tmom.add_Dth('p', m, -1)
        self.tmom.add_term('uph', '+2.0*cos(th[l])', m)
        self.tmom.add_D3sq('uth', m, 'E')
        self.tmom.add_dth('ur', m, 'E/r[k]')
        self.A = self.A + self.tmom.get_coo_matrix()
        del self.tmom

        # Phi-Momentum
        self.add_gov_equation('pmom', 'uph')
        self.pmom.add_dr('bph', m, 'E/Prm*B0[k, l]')
        self.pmom.add_term('uth', '-2.0*cos(th[l])', m)
        self.pmom.add_D3sq('uph', m, 'E')
        self.A = self.A + self.pmom.get_coo_matrix()
        del self.pmom

        ################################
        # Lorentz Equation ##########
        ################################

        # B-divergence replaces r-lorentz
        self.add_gov_equation('bdiv', 'br')
        self.bdiv.add_dr('br', m)
        self.bdiv.add_dth('bth', m)
        self.A = self.A + self.bdiv.get_coo_matrix()
        del self.bdiv

        # theta-Lorentz
        self.add_gov_equation('thlorentz', 'bth')
        self.thlorentz.add_dr('uth', m, 'B0[k, l]')
        self.thlorentz.add_D3sq('bth', m, 'E/Prm')
        self.thlorentz.add_dth('br', m, 'E/Prm/r[k]')
        self.A = self.A + self.thlorentz.get_coo_matrix()
        del self.thlorentz

        # phi-Lorentz
        self.add_gov_equation('phlorentz', 'bph')
        self.phlorentz.add_dr('uph', m, 'B0[k, l]')
        self.phlorentz.add_D3sq('bph', m, 'E/Prm')
        self.A = self.A + self.phlorentz.get_coo_matrix()
        del self.phlorentz

        # Divergence (Mass Conservation) #########
        self.add_gov_equation('div', 'p')
        self.div.add_dr('ur', m)
        self.div.add_dth('uth', m)
        self.A = self.A + self.div.get_coo_matrix()
        del self.div

        # Displacement Equation #########
        self.add_gov_equation('rdisp', 'r_disp')
        self.rdisp.add_term('ur', '1', m)
        self.A = self.A + self.rdisp.get_coo_matrix()
        del self.rdisp

        # Boundary Conditions
        self.add_gov_equation('BC', 'p')

        self.BC.add_bc('ur', 'r[k]**2', 0, m)
        self.BC.add_bc('ur', 'r[k]**2', 0, m, kdiff=1)
        self.BC.add_bc('ur', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('ur', 'r[k]**2', Nk+1, m, kdiff=-1)

        self.BC.add_bc('uth', 'r[k]**2', 0, m)
        self.BC.add_bc('uth', '-r[k]**2', 0, m, kdiff=1)
        self.BC.add_bc('uth', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('uth', '-r[k]**2', Nk+1, m, kdiff=-1)

        self.BC.add_bc('uph', 'r[k]**2', 0, m)
        self.BC.add_bc('uph', '-r[k]**2', 0, m, kdiff=1)
        self.BC.add_bc('uph', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('uph', '-r[k]**2', Nk+1, m, kdiff=-1)

        self.BC.add_bc('p', '1.', 0, m)
        self.BC.add_bc('p', '-1.', 0, m, kdiff=1)
        self.BC.add_bc('p', '1.', Nk+1, m)
        self.BC.add_bc('p', '-1.', Nk+1, m, kdiff=-1)

        self.BC.add_bc('br', 'r[k]**2', 0, m)
        self.BC.add_bc('br', '-r[k]**2', 0, m, kdiff=1)
        self.BC.add_bc('br', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('br', '-r[k]**2', Nk+1, m, kdiff=-1)

        # Free Slip on CMB, Conducting Core at CFB ####
        E = self.E
        Prm = self.Prm
        delta_C = self.delta_C
        r_star = self.r_star
        B0 = self.B0
        dr = self.dr

        self.BC.add_bc('bth', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('bth', 'r[k]**2', Nk+1, m, kdiff=-1)
        for l in range(1, Nl+1):
            row = {'k': 0, 'l': l, 'var': 'bth'}
            self.BC.add_value(str(B0[0, l]/2.), row,
                              {'k': 0, 'l': l, 'var': 'uth'})
            self.BC.add_value(str(B0[1, l]/2.), row,
                              {'k': 1, 'l': l, 'var': 'uth'})
            self.BC.add_value(str(E/Prm*(-1/dr + r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 0, 'l': l, 'var': 'bth'})
            self.BC.add_value(str(E/Prm*(1/dr + r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 1, 'l': l, 'var': 'bth'})

        self.BC.add_bc('bph', 'r[k]**2', Nk+1, m)
        self.BC.add_bc('bph', 'r[k]**2', Nk+1, m, kdiff=-1)
        for l in range(1, Nl+1):
            row = {'k': 0, 'l': l, 'var': 'bph'}
            self.BC.add_value(str(B0[0, l]/2.), row,
                              {'k': 0, 'l': l, 'var': 'uph'})
            self.BC.add_value(str(B0[1, l]/2.), row,
                              {'k': 1, 'l': l, 'var': 'uph'})
            self.BC.add_value(str(E/Prm*(-1/dr + r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 0, 'l': l, 'var': 'bph'})
            self.BC.add_value(str(E/Prm*(1/dr + r_star/(2*delta_C*(1+1j)))),
                              row, {'k': 1, 'l': l, 'var': 'bph'})
        self.A = self.A + self.BC.get_coo_matrix()
        del self.BC

        return self.A

    def make_M(self, m):
        '''
        Creates the M matrix (M*l*x = A*x)
        m: azimuthal fourier mode to compute
        '''

        self.add_gov_equation('B_thlorentz', 'bth')
        self.B_thlorentz.add_term('bth', '1', m)
        self.M = self.B_thlorentz.get_coo_matrix()
        del self.B_thlorentz

        self.add_gov_equation('B_phlorentz', 'bph')
        self.B_phlorentz.add_term('bph', '1', m)
        self.M = self.M + self.B_phlorentz.get_coo_matrix()
        del self.B_phlorentz

        self.add_gov_equation('B_rdisp', 'r_disp')
        self.B_rdisp.add_term('r_disp', '1', m)
        self.M = self.M + self.B_rdisp.get_coo_matrix()
        del self.B_rdisp

        return self.M
