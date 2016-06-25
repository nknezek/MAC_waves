import numpy as np


def apply_d2(model, vec):
    try:
        model.d2Mat
    except:
        model.make_d2Mat()
    return model.d2Mat.tocsr()*vec


def apply_dth(model, vec):
    try:
        model.dthMat
    except:
        model.make_dthMat()
    return model.dthMat.tocsr()*vec


def get_max_d2_norm(model, vec, var=None):
    if var:
        d2_var = model.get_variable(apply_d2(vec), var)
        var_out = model.get_variable(vec, var)
        return abs(d2_var).max()/abs(var_out).max()
    else:
        maxes = []
        d2_vec = apply_d2(model, vec)
        for var in model.model_variables:
            d2_var = model.get_variable(d2_vec, var)
            var_out = model.get_variable(vec, var)
            maxes.append(abs(d2_var).max()/abs(var_out).max())
    return max(maxes)


def get_max_dth_norm(model, vec, var=None):
    if var:
        dth_var = model.get_variable(apply_dth(vec), var)
        var_out = model.get_variable(vec, var)
        return abs(dth_var).max()/abs(var_out).max()
    else:
        maxes = []
        dth_vec = apply_dth(model, vec)
        for var in model.model_variables:
            dth_var = model.get_variable(dth_vec, var)
            var_out = model.get_variable(vec, var)
            maxes.append(abs(dth_var).max()/abs(var_out).max())
    return max(maxes)


def get_equator_power_excess(model, vec, var='ur', split=0.5):
    var_out = model.get_variable(vec, var)
    var_noneq_power = abs(np.concatenate((var_out[:, :(model.Nl-1)*(0.5-split/2.)],
                                         var_out[:, (model.Nl-1)*(0.5+split/2.):]),
                                         axis=1)).sum()
    var_eq_power = abs(var_out[:, (model.Nl-1)*(0.5-split/2.):(model.Nl-1)*(0.5+split/2.)]).sum()
    return var_eq_power-var_noneq_power


def shift_longitude(model, vec, phi):
    return vec*np.exp(1j*model.m_values[0]*phi)


def shift_vec_real(model, vec, var='ur'):
    ''' shift given vector's phase so that given variable (default ur) is
    dominantly real'''
    v = model.get_variable(vec, var)
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
    z = model.get_variable(vec, var)
    ind = np.argmax(np.mean(np.abs(z),axis=1))
    signs = np.sign(z[ind,:])
    return np.where(signs[1:] != signs[:-1])[0]

def get_Q(model, val):
    return abs(val.imag/(2*val.real))

def filter_by_theta_zeros(model, vals, vecs, zeros_wanted, var='uth', verbose=False):
    if type(zeros_wanted) is not list:
        zeros_wanted = list(zeros_wanted)
    if verbose:
        print('zeros wanted: {0}'.format(zeros_wanted))
    filtered_vals = []
    filtered_vecs = []
    for ind,(val, vec) in enumerate(zip(vals, vecs)):
        zc = get_theta_zero_crossings(model, vec, var=var)
        if verbose:
            print("{0}: val= {1}, zc = {2}".format(ind, val, zc))
        if len(zc)-1 in zeros_wanted:
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


def filter_by_d2(model, vals, vecs, max_d2):
    filtered_vals = []
    filtered_vecs = []
    for ind, (val, vec) in enumerate(zip(vals, vecs)):
        if get_max_d2_norm(model, vec) < max_d2:
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