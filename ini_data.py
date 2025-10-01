import numpy as np 


class IniData:
    def __init__(self):
        self.a  = 1     
        self.v0 = 10       
        self.m  = 1  
        self.hbar = 1
        self.e  = 1
        self.unit = False               #True for J, False for eV

    def get_z0(self):
        a    = self.a
        hbar = self.hbar
        m    = self.m
        v0   = self.v0
        e    = self.e

        if self.unit:
            return a/hbar * np.sqrt(2*m*v0)
        else:
            return a/hbar * np.sqrt(2*m*v0*e) 
    
    def get_kappa(self, z):
        z0  = self.get_z0()
        a   = self.a
        return np.sqrt(z0**2 - z**2)/a

    def get_func(self, z):
        z0 = self.get_z0()
        print('Giá trị của z trong get_func =', z)
        if np.abs(z) >= z0:
            raise ValueError('Giá trị không hợp lệ')
        func_even = np.tan(z) - np.sqrt(z0**2 - z**2) / z
        func_odd  = np.tan(z) + z/np.sqrt(z0**2 - z**2)
        # Đạo hàm cho NR
        func_even_deri = 1/(np.cos(z)**2) + z0**2 / (z**2 * np.sqrt(z0**2 - z**2))
        return func_even, func_odd, func_even_deri
    
    def psi_func(self, x , z):
        k = self.get_kappa(z)
        a = self.a

        psi_iw  = np.cos(z*x/a) # wavefunc inwell
        psi_ow  = np.exp(- k * np.abs(x)) # wavefunc outwell
        return psi_iw, psi_ow

if __name__ == '__main__':
    ini = IniData()
    print(ini.get_z0())
    print(ini.get_func(1))