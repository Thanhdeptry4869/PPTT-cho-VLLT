from method import bisection, newton_rapson_hybrid, secant, export_file\
, psi_func_calc
import numpy as np
import ini_data as idt

class Calculator:
    def __init__(self):
        self.ini    = idt.IniData()
        self.a      = self.ini.a
        self.m = self.ini.a
        self.v0 = self.ini.v0
        self.hbar = self.ini.hbar
        self.z0     = self.ini.get_z0()
        self.shift  = 1e-5              # Để tránh những tan(k*pi/2) vô hạn
        self.method_name = 'newton_rapson'  # Default method

    def get_interval(self):
        func    = lambda z: self.ini.get_func(z)[0]  # Even function
        z0      = self.z0
        shift   = self.shift
        n       = int(z0/np.pi*2)
        N       = 200
        print(f'z0 = {z0}, n = {n}')

        interval_k = []         # Chia khoảng theo pi/2
        for i in range(n):
            interval_k.append([i*np.pi/2 + shift, (i+1)*np.pi/2 - shift])
        
        #print(interval_k)
        
        interval_N = []
        for inter in interval_k:
            step = (inter[1] - inter[0]) / N
            start = inter[0]                # reset tại đầu mỗi khoảng
            while start < inter[1]:
                end = start + step
                if end > inter[1]:
                    end = inter[1]
                if func(start) * func(end) < 0:
                    interval_N.append([start, end])
                start = end

                if end >= inter[1]:
                    loop = False
        
        #print(interval_N)

        return interval_N

    def calculate(self, method_name = None, thresh=1e-9, N_max=2000):
        interval = self.get_interval()
        if method_name == None:
            method_name = self.method_name.lower()
        
        # Làm mới file
        with open("results.dat", "w") as f:
            pass  # không ghi gì vào
        with open("results_psi.dat", "w") as f:
            pass  # không ghi gì vào
        with open("loop_results.dat", "w") as f:
            pass  # không ghi gì vào

        for inter in interval:
            print('inter loops = ', inter)
            if method_name == 'bisection':
                p = [inter[0], inter[1]]
                bisection(p, thresh)
            elif method_name == 'newton_rapson':
                #p = [0, (inter[0] + inter[1])/ 2] # p[0] là biến chứa nghiệm update, đặt tùy ý
                newton_rapson_hybrid(inter, thresh, N_max)
            elif method_name == 'secant':
                p = [0, inter[0], inter[1]]
                secant(p, thresh, N_max)
            else:
                raise ValueError('Method not recognized. Use "bisection", "newton_rapson", or "secant".')

        z_values = []

        with open('results.dat', 'r') as f:
            for line in f:
                parts = line.split()
                print(parts)
                if len(parts) >= 2:
                    z_values.append(float(parts[1]))
        
        a   = self.a
        N = 10_000   # Số đoạn chia, chia càng nhiều func càng rõ
        dis = np.linspace(-1.5*a, 1.5*a, N) # Khoảng chạy của x
        count = 0
        for z in z_values:
            count += 1
            export_file('results_psi.dat', z, '\n')
            for x in dis:
                psi_func_calc(x, z)

        print(f'Số z được tính là {count}')

    def Psi_func(self, x , E):
        h = self.hbar
        v0 = self.v0
        m = self.m
        i = 1j
        k = np.sqrt(2*m*E)/h
        l = np.sqrt(2*m*(E + v0))/h
    
        psi_incw1  = np.exp(i*k*x) # wavefunc incoming
        psi_incw2 = np.exp(-i*k*x)
        psi_iw1  = np.sin(l*x) # wavefunc inwell 
        psi_iw2 = np.cos(l*x)
        psi_ow  = np.exp(i*k*x) # wavefunc outcoming
            
        return psi_incw1, psi_incw2, psi_iw1, psi_iw2, psi_ow
        
    def Psi_func_calc(self, x, E):
        a    = self.a
        
        B, C, D, F = self.get_T(E)[0], self.get_T(E)[1], self.get_T(E)[2], self.get_T(E)[3]
        
        if x > a:
            psi = F*self.Psi_func(x, E)[4]
            #psi = 2
            export_file('results_psi.dat', x, psi.real, psi.imag)
            
        elif x < -a:
            psi1 = self.Psi_func(x, E)[0]
            psi2 = B*self.Psi_func(x, E)[1]
            psi = psi1 + psi2
            #psi = 1
            export_file('results_psi.dat', x, psi.real, psi.imag)
        else: 
            psi1 = C*self.Psi_func(x, E)[2]
            psi2 = D*self.Psi_func(x, E)[3]
            psi = psi1 + psi2
            #psi = 0.1
            export_file('results_psi.dat', x, psi.real, psi.imag)
    
    def calculatee(self, E):
        dis = np.linspace(-5*self.a, 5*self.a, 1000)
        for x in dis:
            self.Psi_func_calc(x, E)

    def get_T(self, E):
        a = self.a     
        m = self.m     
        v0 = self.v0      
        hbar = self.hbar    
        with open('results_trans.dat', 'w') as lines:
            pass
        with open('coeffs.dat', 'w') as lines:
            pass

        #for E in range(0,50,5):
        l = np.sqrt(2*m*(E+ self.v0))/hbar
        k = np.sqrt(2*m*E)/hbar
        i = 1j
        
        M = np.array([
            
            [np.exp(i*k*a), np.sin(l*a), -np.cos(l*a), 0],
            
            [-i*k*np.exp(i*k*a), -l*np.cos(l*a), -l*np.sin(l*a), 0],
            
            [0, np.sin(l*a), np.cos(l*a), -np.exp(i*k*a)],
            
            [0, l*np.cos(l*a), -l*np.sin(l*a), -i*k*np.exp(i*k*a)]
        ], dtype=complex)
        
        Y = np.array([
            -np.exp(-i*k*a),
            -i*k*np.exp(-i*k*a),
            0,
            0
        ], dtype=complex)
            
        try:
            X = np.linalg.solve(M, Y)
            B, C, D, F = X[0], X[1], X[2], X[3]
            with open('coeffs.dat', 'a') as lines:
                lines.write(f'1 \t\t {B} \t\t {C} \t\t {D} \t\t {F} \n')
            #print("-" * 40)
            R_coeff_sq = np.abs(B)**2
            T_coeff_sq = np.abs(F)**2
            '''
            print(f"|R|^2 = {R_coeff_sq:.6f}")
            print(f"|T|^2 = {T_coeff_sq:.6f}")
            '''
            print('---------')
            with open('results_trans.dat', 'a') as lines:
                lines.write(f'{E} \t\t {T_coeff_sq} \n')
            #print(f"--- Kết quả Giải Hệ số Tán xạ cho k={} (A=1) ---")
            '''
            print(f"Hệ số Phản xạ B: {B:.6f}")
            print(f"Hệ số C: {C:.6f}")
            print(f"Hệ số D: {D:.6f}")
            print(f"Hệ số Truyền qua F: {F:.6f}")
            '''
            return B, C, D, F
        except np.linalg.LinAlgError:
            print("Cảnh báo: Ma trận không khả nghịch (Singular Matrix). Không thể giải.")
            return np.nan + i*np.nan, np.nan + i*np.nan, np.nan + i*np.nan, np.nan + i*np.nan
            
                   

        
if __name__ == '__main__':
    cal = Calculator()
    # interval = cal.get_interval()
    #cal.calculate(method_name='bisection')
    cal.calculatee(E = 1)
