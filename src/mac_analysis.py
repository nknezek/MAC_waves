import numpy as np


def apply_D3sq(model, vec):
    try:
        model.D3sqMat
    except:
        model.make_D3sqMat()
    return model.D3sqMat.tocsr()*vec


def apply_dth(model, vec):
    try:
        model.dthMat
    except:
        model.make_dthMat()
    return model.dthMat.tocsr()*vec


def get_max_D3sq_norm(model, vec, var=None):
    if var:
        D3sq_var = model.get_variable(apply_D3sq(vec), var, returnBC=False)
        var_out = model.get_variable(vec, var, returnBC=False)
        return abs(D3sq_var).max()/abs(var_out).max()
    else:
        maxes = []
        D3sq_vec = apply_D3sq(model, vec)
        for var in model.model_variables:
            D3sq_var = model.get_variable(D3sq_vec, var, returnBC=False)
            var_out = model.get_variable(vec, var, returnBC=False)
            maxes.append(abs(D3sq_var).max()/abs(var_out).max())
    return max(maxes)


def get_max_dth_norm(model, vec, var=None):
    if var:
        dth_var = model.get_variable(apply_dth(vec), var, returnBC=False)
        var_out = model.get_variable(vec, var, returnBC=False)
        return abs(dth_var).max()/abs(var_out).max()
    else:
        maxes = []
        dth_vec = apply_dth(model, vec)
        for var in model.model_variables:
            dth_var = model.get_variable(dth_vec, var, returnBC=False)
            var_out = model.get_variable(vec, var, returnBC=False)
            maxes.append(abs(dth_var).max()/abs(var_out).max())
    return max(maxes)


def get_equator_power_excess(model, vec, var='ur', split=0.5):
    var_out = model.get_variable(vec, var, returnBC=False)
    var_noneq_power = abs(np.concatenate((var_out[:, :model.Nl*(0.5-split/2.)],
                                         var_out[:, model.Nl*(0.5+split/2.):]),
                                         axis=1)).sum()
    var_eq_power = abs(var_out[:, model.Nl*(0.5-split/2.):model.Nl*(0.5+split/2.)]).sum()
    return var_eq_power-var_noneq_power


def shift_longitude(model, vec, phi):
    return vec*np.exp(1j*model.m_values[0]*phi)


def shift_vec_real(model, vec, var='ur'):
    ''' shift given vector's phase so that given variable (default ur) is
    dominantly real'''
    v = model.get_variable(vec, var, returnBC=False)
    angs = np.angle(v) % np.pi
    abs_v = np.abs(v)
    avg_ang = np.average(angs, weights=abs_v) # shift phase angle
    # shift case to deal with vectors that are already dominantly real
    var_ang = np.average((angs - avg_ang)**2, weights=abs_v)
    if var_ang > 0.5:
        shift = np.exp(0.5j*np.pi)
        v_s = v*shift
        angs_s = np.angle(v_s) % np.pi
        avg_ang = np.average(angs_s, weights=abs_v)
        return vec*np.exp(-1j*(avg_ang-0.5*np.pi))
    else:
        return vec*np.exp(-1j*avg_ang)

def get_theta_zero_crossings(model, vec, var='uth'):
    uth = model.get_variable(vec, var, returnBC=False)
    signs = np.sign(np.diff(np.mean(np.abs(uth), axis=0)))
    return np.where(signs[1:] != signs[:-1])[0]

def get_Q(model, val):
    return abs(val.imag/(2*val.real))

def filter_by_theta_zeros(model, vals, vecs, zeros_wanted, val='uth'):
    if type(zeros_wanted) is not list:
        zeros_wanted = [zeros_wanted]
    filtered_vals = []
    filtered_vecs = []
    for (val, vec) in zip(vals, vecs):
        if len(get_theta_zero_crossings(model, vec))-1 in zeros_wanted:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs

def filter_by_dth(model, vals, vecs, max_dth):
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        if get_max_dth_norm(model, vec) < max_dth:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs


def filter_by_D3sq(model, vals, vecs, max_D3sq):
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        if get_max_D3sq_norm(model, vec) < max_D3sq:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs

def filter_by_Q(model, vals, vecs, min_Q):
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        if get_Q(model, val) > min_Q:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs

def filter_by_equator_power(model, vals, vecs, equator_fraction=0.5, var='ur'):
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        if get_equator_power_excess(model, vec, var=var,
                                    split=equator_fraction) > 0.:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs

def filter_by_period(model, vals, vecs, minT, maxT):
    '''
    filter results by wave period in years 
    '''
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        Period = (2*np.pi/val.imag)*model.t_star/(24.*3600.*365.25)
        if Period > minT and Period < maxT:
            filtered_vals.append(val)
            filtered_vecs.append(vec)
    return filtered_vals, filtered_vecs

def get_period(model, val):
    return 2*np.pi/val*model.t_star/(24.*3600.*365.25)