import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

class waveguideModel:
    #__slots__ = 'width', 'length', 'wg_model', 'loss'
    def __init__(self, optical_model, width, length, loss) -> None:
        self.optical = optical_model
        self.width = width
        self.length = length
        self.loss = loss
        return

    @property
    def alpha(self):
        return math.log(10)*self.loss*100

    def neff(self, wl) -> float:
        return self.optical.neff(wl)

    def tm(self, wl):

        def homogeneous_wg_mat(wl, n_eff_i, length, loss):
            beta = 2*np.pi*n_eff_i/wl - 1j * loss/2
            hwg_mat = np.array([
                ( np.exp( 1j * beta * length) , 0),
                ( 0 , np.exp( -1j * beta * length ) )
            ],
            dtype = complex
            )
            return hwg_mat

        cavity_length = self.length
        n_eff = self.neff( wl )

        hwg_mat_1 = homogeneous_wg_mat( wl, n_eff, cavity_length, self.alpha )

        return hwg_mat_1

class braggUnitGeometry:
    __slots__ = 'period', 'width', 'grating_height'
    def __init__(self, period, width, grating_height) -> None:
        self.period = period
        self.width = width
        self.grating_height = grating_height
        return

    @property
    def aspect_ratio(self) -> float:
        return 2 * self.grating_height / self.period

class waveGuideCompact:
    __slots__ = 'n1', 'n2', 'n3', 'lambda_0'
    def __init__(self, *args, **kwargs) -> None:
        if args is not None:
            if len(args) == 4:
                self.n1 = args[0]
                self.n2 = args[1]
                self.n3 = args[2]
                self.lambda_0 = args[3]
            else:
                raise ValueError("Wrong number of arguments passed to wgCompact")
        return

    def neff(self, wl = None) -> float:
        if wl is None:
            wl = self.lambda_0
        return self.n1 + self.n2 * (wl * 1e6 - self.lambda_0 * 1e6) + self.n3 * (wl * 1e6 - self.lambda_0 * 1e6) ** 2

class braggUnitOptical:
    __slots__ = 'wg_model' , 'bragg_wl', 'bandwidth', 'kappa'
    def __init__(self, wg_compact, bandwidth, kappa) -> None:
        self.wg_model = wg_compact
        self.bragg_wl = self.wg_model.lambda_0
        self.bandwidth = bandwidth
        self.kappa = kappa
        return

    def neff(self, wl) -> float:
        return self.wg_model.neff( wl )

class braggUnitCell:
    __slots__ = 'optical', 'geometry', 'loss'
    def __init__(self, unit_optical_model, unit_geometry_model, loss ) -> None:
        if not ( isinstance(unit_optical_model, braggUnitOptical) and isinstance(unit_geometry_model, braggUnitGeometry)):
            raise ValueError("")
        self.optical = unit_optical_model
        self.geometry = unit_geometry_model
        self.loss = loss
        return

    @property
    def alpha( self ) -> float:
        return math.log(10) * self.loss * 100
    
    def tm(self, wl : float ) :
        def homogeneous_wg_mat(wl : float, n_eff_i : float, length : float, loss : float):
            beta = 2 * np.pi * n_eff_i / wl - 1j * loss / 2
            hwg_mat = np.array([
                ( np.exp( 1j * beta * length) , 0),
                ( 0 , np.exp( -1j * beta * length ) )
            ],
            dtype = complex
            )
            return hwg_mat
        
        def index_step_mat(neff_1, neff_2):
            a = (neff_1 + neff_2)/(2 * np.sqrt( neff_1 * neff_2 ))
            b = (neff_1 - neff_2)/(2 * np.sqrt( neff_1 * neff_2 ))
            is_mat = np.array([
                (a , b),
                (b , a)
            ],
            dtype = complex
            )
            return is_mat

        #delta_n = 0.083803860069 
        delta_n = self.optical.kappa * self.optical.bragg_wl / 2
        cavity_length = self.geometry.period / 2
        n_eff_1 = self.optical.neff(wl) - delta_n / 2
        n_eff_2 = self.optical.neff(wl) + delta_n / 2

        hwg_mat_1 = homogeneous_wg_mat(wl, n_eff_1, cavity_length, self.alpha )
        is_mat_12 = index_step_mat(n_eff_1, n_eff_2 )

        hwg_mat_2 = homogeneous_wg_mat(wl, n_eff_2, cavity_length, self.alpha )
        is_mat_21 = index_step_mat( n_eff_1, n_eff_2 )

        tm_1 = np.matmul( hwg_mat_1, is_mat_12 )
        tm_2 = np.matmul( hwg_mat_2, is_mat_21 )

        tm = np.matmul( tm_1, tm_2 )

        return tm

class braggGrating:
    #__slots__ = 'unit_cell', 'grating_count'
    def __init__(self, bragg_unit_cell, grating_count) -> None:
        if not isinstance(bragg_unit_cell, braggUnitCell):
            raise ValueError("Wrong type of object passed to braggGrating")
        self.unit_cell = bragg_unit_cell
        self.grating_count = grating_count

    def tm(self, wl):
        return np.linalg.matrix_power( self.unit_cell.tm(wl), self.grating_count)

    def tr_set(self, wl):
        tm = self.tm( wl )
        t = np.absolute( 1 / tm[0][0] ) ** 2
        r = np.absolute( tm[1][0] / tm[0][0] ) ** 2
        return t,r

class cell:
    def __init__(self, *args) -> None:
        self.sub_cells = list()
        for arg in args:
            self.sub_cells.append( arg )
        return

    def tm(self,wl):
        tm = None
        for sub_cell in self.sub_cells:
            if tm is None:
                tm = sub_cell.tm(wl)
            else:
                tm = np.matmul( tm, sub_cell.tm(wl) )
        return tm

    def tr_set(self, wl):
        tm_circ = self.tm( wl )
        t = np.absolute( 1 / tm_circ[0][0] ) ** 2
        r = np.absolute( tm_circ[1][0] / tm_circ[0][0] ) ** 2
        return t,r

def sweep_bragg_wl(bragg, start_wl, stop_wl, resolution,t):
    if not (isinstance(bragg, braggGrating) or isinstance(bragg, cell)):
        raise ValueError()
    point_count = round((stop_wl - start_wl)*1e9/resolution) 
    wl_set = np.linspace(start_wl, stop_wl, point_count)
    tr_set = np.asarray([ bragg.tr_set( wl ) for wl in wl_set ])

    t_set = tr_set[ : , 0]
    r_set = tr_set[ : , 1]

    max_r = np.amax(r_set)
    max_wl = wl_set[np.where(r_set == max_r)]*1e9

    wl_set_nm = wl_set * 1e9



    fig,ax = plt.subplots(2)
    #fig.suptitle(f'width [nm]={round(bragg.unit_cell.geometry.width * 1e9)},n1={bragg.unit_cell.optical.wg_model.n1},n2={bragg.unit_cell.optical.wg_model.n2},n3{bragg.unit_cell.optical.wg_model.n3}')
    fig.suptitle(f"f={t}")
    ax[0].plot(wl_set_nm, t_set, label='Transmittion')
    ax[0].plot(wl_set_nm, r_set, label='Reflection')
    #ax[0].axvline(max_wl)
    ax[0].set(ylabel = 'Response [%]')

    ax[1].plot(wl_set_nm, 10*np.log(t_set), label='Transmittion')
    ax[1].plot(wl_set_nm, 10*np.log(r_set), label='Reflection')
    #ax[1].axvline(max_wl)
    ax[1].set(xlabel = 'Wavelength $\lambda$ [nm]',ylabel = 'Response [dB]')

    plt.show(block=True)
    return ax

def main():
    #n1,n2,n3 = 2.40559,-1.8643,0.769594
    #n1,n2,n3 = 2.47066,-1.79948,0.663474
    #n1,n2,n3 = 2.45667,-1.8226,0.693588
    n1,n2,n3 = 2.4397,-1.84213,0.723176

    lam_0 = 1310e-9
    wg_comp = waveGuideCompact(n1,n2,n3,lam_0)

    bragg_period = 268e-9 #*2
    bragg_width = 350e-9
    bragg_grating_height = dw = 45e-9
    bragg_geo = braggUnitGeometry( bragg_period, bragg_width, bragg_grating_height )
    
    bragg_kappa = -1.53519e19 * dw ** 2 + 2.2751e12 * dw
    #print(f'{bragg_kappa=}')
    bragg_kappa = 130000
    bragg_bandwidth = 1e-9
    bragg_opt = braggUnitOptical(wg_comp, bragg_bandwidth, bragg_kappa)

    loss = 3
    bragg_unit = braggUnitCell(bragg_opt, bragg_geo, loss)

    period_count = 800
    bragg = braggGrating( bragg_unit, period_count )

    fp = waveguideModel(bragg_opt, bragg_width, 1110e-6, 0)
    bend = waveguideModel(bragg_opt, bragg_width, bragg_period , loss)

    circuit = cell( bragg, fp, bragg)

    span = 5e-9
    c_wl = 1310e-9
    start = c_wl - span/2
    stop = c_wl + span/2
    res = 0.01

    '''
    window = 100e-6
    base = 1200e-6
    for x in np.linspace(base-window,base + window,30):
        fp.length = x
        sweep_bragg_wl(circuit,start,stop,res,f'{x}')
    '''

    '''
    window = 5e-9
    base = 267e-9
    for x in np.linspace(base-window,base + window,10):
        bragg_geo.period = x
        fp.width = x/2
        sweep_bragg_wl(circuit,start,stop,res,f'{x}')
    '''
    '''
    window = 25e-9
    base = 25e-9
    for x in np.linspace(base-window,base + window,20):
        bragg_geo.grating_height = x
        #bragg_opt.kappa = -1.53519e19 * x ** 2 + 2.2751e12 * x
        sweep_bragg_wl(circuit,start,stop,res,f'{x}')
    '''
    for x in np.linspace(100,1000,20):
        bragg.grating_count = round(x)
        sweep_bragg_wl(circuit,start,stop,res,f'{x}')
    '''
    '''
    '''
    for x in np.linspace(1,20,10):
        bragg_unit.loss = x
        sweep_bragg_wl(circuit,start,stop,res,f'{x}')
    '''

    #sweep_bragg_wl(circuit,start,stop,res,f'{3}')
    

    return

if __name__ == '__main__':
    main()
