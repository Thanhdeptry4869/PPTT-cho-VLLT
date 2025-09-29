import numpy as np 


class IniData:
    def __init__(self):
        self.a  = 1e-9
        self.v0 = 50
        self.m  = 9.11e-31
        self.hbar = 1.054e-34
        self.e  = 1.6e-19
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
        
    def get_func(self, z):
        z0 = self.get_z0()
        func_even = np.tan(z) - np.sqrt(z0**2 - z**2) / z
        func_odd  = np.tan(z) + z/np.sqrt(z0**2 - z**2)
        return func_even, func_odd
    

if __name__ == '__main__':
    ini = IniData()
    print(ini.get_z0())
    print(ini.get_func(1))