from method import bisection, newton_rapson_hybrid, secant, export_file\
, psi_func
import numpy as np
import ini_data as idt

class Calculator:
    def __init__(self):
        self.ini    = idt.IniData()
        self.a      = self.ini.a
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
        N = 1000   # Số đoạn chia, chia càng nhiều func càng rõ
        dis = np.linspace(-10*a, 10*a, N) # Khoảng chạy của x
        count = 0
        for z in z_values:
            count += 1
            #export_file('results_psi.dat', z, '\n')
            for x in dis:
                psi_func(x, z)

        print(f'Số z được tính là {count}')
            
if __name__ == '__main__':
    cal = Calculator()
    # interval = cal.get_interval()
    cal.calculate(method_name='secant')
