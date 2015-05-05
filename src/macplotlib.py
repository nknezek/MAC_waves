

##### Display and Save 1D Waves with comparison Spherical Harmonics for found Eigenvalues and vector field map ######
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from numpy import sin
from numpy import cos
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib as mpl

colors = ['b','g','r','m','y','k','c']
def plot_1D(model,vec,val,m,l):
    E = model.E
    Nk = model.Nk
    Nl = model.Nl
    th = model.th


    ## Plot Figures
    fig, axes = plt.subplots(3,1,figsize=(15,8), sharex=True, sharey=True)
    titles = ['Absolute Value','Real','Imaginary']
    for (ax,title,ind) in zip(axes,titles,range(len(axes))):
        ax.set_title(title)
        line = []
        for (var,color) in zip(model.model_variables,colors):
             (out,bbound,tbound)=model.get_variable(vec,var)
             if ind == 0:
                  line.append(ax.plot(th[1:-1]*180./np.pi,abs(out.T),color=color))
             elif ind ==1:
                  ax.plot(th[1:-1]*180./np.pi,out.T.real,color=color)
                  ax.grid()
             elif ind==2:
                  ax.plot(th[1:-1]*180./np.pi,out.T.imag,color=color)
                  ax.grid()
        if ind ==0:
             labels = ['ur','uth','uph','br','bth','bph','p']
             ax.legend([x[0] for x in line],labels,loc=0,ncol=4)
             ax.grid()

    plt.suptitle('MAC, Eigenvalue = {0:.5f}, m={1}, l={2}\n Nk={3}, Nl={4}, E={5:.2e})'.format(val,m,l,Nk,Nl,E), size=14)
    plt.savefig('./output/m={1}/MAC_Eig{0:.2f}j_m={1}_l={2}_Nk={3}_Nl={4}_E={5:.2e}.png'.format(val.imag,m,l,Nk,Nl,E))


def plot_mollyweide(model,vec,val,m,l):
    E = model.E
    Nk = model.Nk
    Nl = model.Nl
    th = model.th

    ## Calculate vector field and contour field for plotting with basemap
    ## Create full vector grid in theta and phi
    u_1D = model.get_variable(vec,'uph')[0][0]
    v_1D = model.get_variable(vec,'uth')[0][0]
    Nph = 2*Nl
    ph = np.linspace(-180.,180.-360./Nph,Nph)
    lon_grid, lat_grid = np.meshgrid(ph,th[1:-1]*180./np.pi-90.,)
    v = (np.exp(1j*m*lon_grid*np.pi/180.).T*v_1D).T
    u = (np.exp(1j*m*lon_grid*np.pi/180.).T*u_1D).T
    absu=u**2 + v**2
    Nvec = np.floor(Nl/20.)

    ### Plot Mollweide Projection
    plt.figure(figsize=(10,10))
    ## Set up map
    bmap = Basemap(projection='moll',lon_0=0.)
    bmap.drawparallels(np.arange(-90.,90.,15.))
    bmap.drawmeridians(np.arange(0.,360.,15.))
    ## Convert Coordinates to those used by basemap to plot
    lon,lat = bmap(lon_grid,lat_grid)
    bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
    bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
    plt.title('MAC, Mollweide Projection Vector Field for m={0}, l={1}'.format(m,l))
    plt.savefig('./output/m={1}/MAC_MollweideVectorField_m={1}_l={2}.png'.format(val.imag,m,l))

def plot_robinson(model,vec,val,m,l):
    E = model.E
    Nk = model.Nk
    Nl = model.Nl
    th = model.th

    ## Plot Robinson vector field
    #### Display waves on a Spherical Map Projection
    projtype = 'robin'

    ## Create full vector grid in theta and phi
    u_1D = model.get_variable(vec,'uph')[0][0]
    v_1D = model.get_variable(vec,'uth')[0][0]
    Nph = 2*Nl
    ph = np.linspace(-180.,180.-360./Nph,Nph)
    lon_grid, lat_grid = np.meshgrid(ph,th[1:-1]*180./np.pi-90.,)
    v = (np.exp(1j*m*lon_grid*np.pi/180.).T*v_1D).T
    u = (np.exp(1j*m*lon_grid*np.pi/180.).T*u_1D).T
    absu=u**2 + v**2
    Nvec = np.floor(Nl/20.)

    ### Plot Robinson Projection
    plt.figure(figsize=(10,10))
    ## Set up map
    bmap = Basemap(projection=projtype,lon_0=0.)
    bmap.drawparallels(np.arange(-90.,90.,15.))
    bmap.drawmeridians(np.arange(0.,360.,15.))
    ## Convert Coordinates to those used by basemap to plot
    lon,lat = bmap(lon_grid,lat_grid)
    bmap.contourf(lon,lat,absu,15,cmap=plt.cm.Reds,alpha=0.5)
    bmap.quiver(lon[::Nvec,::Nvec],lat[::Nvec,::Nvec],u[::Nvec,::Nvec].real,v[::Nvec,::Nvec].real)
    plt.title('MAC {0} Projection Vector Field for m={1}, l={2}'.format('Robinson',m,l))
    plt.savefig('./output/m={1}/MAC_{0}VectorField_m={1}_l={2}.png'.format('Robinson',m,l))

def plot_A(A,m):
    ### Plot A Matrix (M*l*x = A*x) ###

    plt.figure(figsize=(10,10))
    plt.spy(np.abs(A.todense()))
    plt.grid()
    plt.title('A,m='+str(m)+' matrix all terms (M*l*x = A*x)')
    plt.savefig('./output/m={0}/A_matrix_m={0}.png'.format(m))

    # plt.subplot(3,2,2)
#   plt.spy(np.abs(A.todense().real))
#   plt.grid()
#   plt.title('A'+str(m)+' real')
#
#   plt.subplot(3,2,3)
#   plt.spy(np.abs(A.todense().imag))
#   plt.grid()
#   plt.title('A'+str(m)+' imaginary')
#
#   A_pos = np.matrix(A.todense())
#   A_pos[A.todense()<0.] = 0
#   plt.subplot(3,2,4)
#   plt.spy(np.abs(A_pos))
#   plt.grid()
#   plt.title('A'+str(m)+' positive')
#
#   A_neg = np.matrix(A.todense())
#   A_neg[A.todense()>0.] = 0
#   plt.subplot(3,2,5)
#   plt.spy(np.abs(A_neg))
#   plt.grid()
#   plt.title('A'+str(m)+' negative')
#   plt.tight_layout()
#   plt.savefig('./output/m={0}/A_matrix_m={0}.png'.format(m))

def plot_M(M,m):
    plt.figure(figsize=(10,10))
    plt.title('M Matrix (M*l*x = A*x)')
    plt.spy(np.abs(M.todense()))
    plt.grid()
    plt.savefig('./output/m={0}/M_matrix_m={0}.png'.format(m))

def plot_pcolor_rth(model,val,vec,dir_name='./',title='pcolor MAC Wave Plot'):
    plt.close('all')
    r_star = model.r_star
    t_star = model.t_star
    P_star = model.P_star
    B_star = model.B_star
    u_star = model.u_star

    rpl = model.r[1:]*r_star
    thpl = model.th[1:-1]*180./np.pi

    fig = plt.figure(figsize=(14,14))
    fig.suptitle(title)
    ur,urtop,urbottom = model.get_variable(vec,'ur')
    uth,uthtop,uthbottom = model.get_variable(vec,'uth')
    uph,uphtop,uphbottom = model.get_variable(vec,'uph')
    ur = ur*u_star
    uth = uth*u_star
    uph = uph*u_star

    urmax = np.amax(abs(ur))
    plt.subplot(8,2,1)
    plt.pcolor(thpl,rpl,ur.real, cmap='RdBu',vmin=-urmax, vmax=urmax)
    plt.title('ur real (m/s)')
    plt.subplot(8,2,2)
    plt.pcolor(thpl,rpl,ur.imag, cmap='RdBu',vmin=-urmax, vmax=urmax)
    plt.title('ur imag (m/s)')
    plt.colorbar()

    uthmax = np.amax(abs(uth))
    plt.subplot(8,2,3)
    plt.pcolor(thpl,rpl,uth.real, cmap='RdBu',vmin=-uthmax, vmax=uthmax)
    plt.title('uth real (m/s)')
    plt.subplot(8,2,4)
    plt.pcolor(thpl,rpl,uth.imag, cmap='RdBu',vmin=-uthmax, vmax=uthmax)
    plt.title('uth imag (m/s)')
    plt.colorbar()

    uphmax = np.amax(abs(uph))
    plt.subplot(8,2,5)
    plt.pcolor(thpl,rpl,uph.real, cmap='RdBu',vmin=-uphmax, vmax=uphmax)
    plt.title('uph real (m/s)')
    plt.subplot(8,2,6)
    plt.pcolor(thpl,rpl,uph.imag, cmap='RdBu',vmin=-uphmax, vmax=uphmax)
    plt.title('uph imag (m/s)')
    plt.colorbar()

    br,brtop,brbottom = model.get_variable(vec,'br')
    bth,bthtop,bthbottom = model.get_variable(vec,'bth')
    bph,bphtop,bphbottom = model.get_variable(vec,'bph')
    br = br*B_star
    bth = bth*B_star
    bph = bph*B_star

    brmax = np.amax(abs(br))
    plt.subplot(8,2,7)
    plt.pcolor(thpl,rpl,br.real, cmap='RdBu',vmin=-brmax, vmax=brmax)
    plt.title('br real (T)')
    plt.subplot(8,2,8)
    plt.pcolor(thpl,rpl,br.imag, cmap='RdBu',vmin=-brmax, vmax=brmax)
    plt.title('br imag (T)')
    plt.colorbar()

    bthmax = np.amax(abs(bth))
    plt.subplot(8,2,9)
    plt.pcolor(thpl,rpl,bth.real, cmap='RdBu',vmin=-bthmax, vmax=bthmax)
    plt.title('bth real (T)')
    plt.subplot(8,2,10)
    plt.pcolor(thpl,rpl,bth.imag, cmap='RdBu',vmin=-bthmax, vmax=bthmax)
    plt.title('bth imag (T)')
    plt.colorbar()

    bphmax = np.amax(abs(bph))
    plt.subplot(8,2,11)
    plt.pcolor(thpl,rpl,bph.real, cmap='RdBu',vmin=-bphmax, vmax=bphmax)
    plt.title('bph real (T)')
    plt.subplot(8,2,12)
    plt.pcolor(thpl,rpl,bph.imag, cmap='RdBu',vmin=-bphmax, vmax=bphmax)
    plt.title('bph imag (T)')
    plt.colorbar()

    p,ptop,pbottom = model.get_variable(vec,'p')
    r_disp = model.get_variable(vec,'r_disp')
    p = p*P_star
    r_disp = r_disp*r_star

    pmax = np.amax(abs(p))
    plt.subplot(8,2,13)
    plt.pcolor(thpl,rpl,p.real, cmap='RdBu',vmin=-pmax, vmax=pmax)
    plt.title('p real (Pa)')
    plt.subplot(8,2,14)
    plt.pcolor(thpl,rpl,p.imag, cmap='RdBu',vmin=-pmax, vmax=pmax)
    plt.title('p imag (Pa)')
    plt.colorbar()

    r_dispmax = np.amax(abs(r_disp))
    plt.subplot(8,2,15)
    plt.pcolor(thpl,rpl,r_disp.real, cmap='RdBu',vmin=-r_dispmax, vmax=r_dispmax)
    plt.title('r_dispmax real (m)')
    plt.subplot(8,2,16)
    plt.pcolor(thpl,rpl,r_disp.imag, cmap='RdBu',vmin=-r_dispmax, vmax=r_dispmax)
    plt.title('r_dispmax imag (m)')
    plt.colorbar()

    fig.tight_layout()
    plt.savefig(dir_name+title+'.png')
