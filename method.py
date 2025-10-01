import numpy as np

def bisection(dis, thresh):     
    diff        = thresh + 1
    threshold   = 1e-5          #Mức sai số chấp nhận
    N = 0
    #export_file(f'Bisection Method','\n','-'*140,'\n','-'*140,'\n','N', 'a', 'b', 'ref', 'f(ref)','\n', '-'*140, header=True)
    while diff > thresh:
        N += 1
        ref = (dis[0] + dis[1])/2

        export_file('loop_results.dat', N, dis[0], dis[1], ref, func(ref)[0])

        if func(dis[0])[0]*func(dis[1])[0] > 0:
            raise ValueError("Khoảng không chứa nghiệm (cùng dấu).")


        if np.abs(func(ref)[0]) < threshold:
            print(f'The result is {ref}')
            export_file('results.dat', N, dis[0], dis[1], ref, func(ref)[0])
            print(f"Nghiệm xấp xỉ = {ref:.12f}, f(ref) = {func(ref)[0]:.3e}")
            #export_file('results.dat', '-'*140)
            break
        elif func(dis[0])[0] * func(ref)[0] < 0:
            dis[1] = ref
        elif func(ref)[0] * func(dis[1])[0] < 0:
            dis[0] = ref
        else:
            print('-'*30)
            print(f'This distance does not have a root, after {N} iterations')
            print('-'*30)
            raise ValueError
        
        diff = np.abs(dis[0] - dis[1])
        # if np.abs(dis[0] - dis[1]) < thresh and np.abs(func((dis[0] + dis[1])/2)) < thresh:
        #     print('The result is {ans} after {N} iterations'.format(ans = (dis[0] + dis[1])/2, N = N))
        #     #print('diff  = ', diff)


    export_file('loop_results.dat', '-'*140)
    #export_file('results.dat', '-'*140)

def fixed_point(p, thresh, N_max):
    N = 0
    export_file(f'Fixed Point Method ','\n','-'*140,'\n','-'*140,'\n','N', '1', '2', '3', '4', '5', header=True)
    n_func = len(func(1, method=False))
    iter = True

    diff = np.zeros(n_func) + thresh + 1
    values  = np.asarray(func(p[1], method=False))
    pack = np.zeros(n_func) + p[1] 
    limit_point = 1e+05


    while iter:
        #values  = func(p, method=False)
        garb  = []

        for i in range(n_func):
            #values[i] = func(pack[i], method=False)[i]
            diff[i] = np.abs(values[i] - pack[i])

            if diff[i] < thresh:
                #print('The result of func {i} is {ans} after {N} iterations'.format(i = i, ans = values[i], N = N)  )
                if np.abs(values[i]) > limit_point:
                    garb.append(0)
                else:
                    garb.append(values[i])
                #continue
            else:
                pack[i] = values[i]
                values[i] = func(pack[i], method=False)[i]
                if np.abs(values[i]) > limit_point:
                    garb.append(0)
                else:
                    garb.append(values[i])

    
        export_file(N, *garb)
        N += 1
        if N == N_max:
            #print('-'*30)
            #print(f'Function {i} does not converge after {N} iterations')
            #print('-'*30)
            break
    export_file('-'*140)


def newton_rapson(inter, thresh, N_max):
    diff = thresh + 1
    N = 0
    p = [0, (inter[0] + inter[1])/2]
    while diff > thresh:
        p[0] = p[1] - func(p[1])[0] / func(p[1])[1]
        diff = np.abs(p[0] - p[1])
        export_file('loop_results.dat', N, p[0], func(p[0])[0])
        if diff < thresh:
            print('The result is {ans} after {N} iterations'.format(ans = p[0], N = N))
            break
        else:
            p[1] = p[0]
        N += 1
        if N == N_max:
            print('-'*30)
            print(f'This distance does not converge after {N} iterations')
            print('-'*30)
            raise ValueError
        
def newton_rapson_hybrid(inter, thresh, N_max = None):
    diff = thresh + 1
    N    = 0
    p = [0, (inter[0] + inter[1])/2]
    while diff > thresh:
        p[0] = p[1] - func(p[1])[0] / func(p[1])[1]
        diff = np.abs(p[0] - p[1])
        export_file('loop_results.dat', N, p[0], func(p[0])[0], diff)
        if diff < thresh or np.abs(inter[0] - inter[1]) < thresh:
            export_file('results.dat', N, p[0], func(p[0])[0], diff)
            print('The result is {ans} after {N} iterations'.format(ans = p[0], N = N))
            break
        elif p[0] < inter[0] or p[0] > inter[1]:
            p[1] = (inter[0] + inter[1]) /2
        else:
            p[1] = p[0]

        if func(inter[0])[0] * func(p[1])[0] < 0:
            inter[1] = p[1]
        elif func(inter[1])[0] * func(p[1])[0] < 0:
            inter[0] = p[1] 
        else:
            raise ValueError('Newton_Bisec có vấn đề!!')
        

        N += 1
        if N == N_max:
            print('-'*30)
            print(f'This distance does not converge after {N} iterations')
            print('-'*30)
            #raise ValueError



def secant(p, thresh, N_max):
    diff = thresh + 1
    N = 0
    while diff > thresh:
        p[0] = p[1] - func(p[1])[0] * (p[1] - p[2]) / (func(p[1])[0] - func(p[2])[0])
        diff = np.abs(p[0] - p[1])
        export_file('loop_results.dat', N, p[0], func(p[0])[0], diff)
        if diff < thresh:
            export_file('results.dat', N, p[0], func(p[0])[0], diff)
            print('The result is {ans} after {N} iterations'.format(ans = p[0], N = N))
            break
        else:
            p[1], p[2] = p[0], p[1]
        N += 1
        if N == N_max:
            print('-'*30)
            print(f'This distance does not converge after {N} iterations')
            print('-'*30)
            raise ValueError


from ini_data import IniData
ini = IniData()

def func(z):
    # from ini_data import IniData
    # ini = IniData()
    return ini.get_func(z)[0], ini.get_func(z)[1]

def kappa(z):
    # from ini_data import IniData
    # ini = IniData()
    return ini.get_kappa(z)

##### Now, we need to normalize some coefficients in Psi ##########

def norm_func(z):
    from scipy.integrate import quad

    # from ini_data import IniData
    # ini = IniData()
    a   = ini.a
    psi_iw  = lambda x: ini.psi_func(x, z)[0]**2
    psi_ow  = lambda x: ini.psi_func(x, z)[1]**2

    iw, _ = quad(psi_iw, -a, a)        # Kết quả tích phân bên trong giếng
    ow, _ = quad(psi_ow, a, np.inf)    # Kết quả tích phân bên ngoài giếng

    # |F|^2 = const * |D|^2
    const = np.cos(z)**2 / np.exp(-2 * kappa(z) * a)
    
    # Thế vào pt chuẩn hóa
    D = np.sqrt(1/(iw + 2*const*ow))
    F = np.sqrt(D*D*const)
    return D, F

def psi_func(x, z):
    # from ini_data import IniData
    # ini = IniData()
    a    = ini.a
    l    = z/a

    D, F = norm_func(z)

    if np.abs(x) > a:
        psi = F*np.exp(- kappa(z) * x)
        export_file('results_psi.dat', x, psi)
        #return F*np.exp(- kappa(z) * x)
    else: 
        psi = D*np.cos(l*x)
        export_file('results_psi.dat', x, psi)
        #return D*np.cos(l*x)

def export_file(file_name, *args, header=False):
    file = open(file_name, 'a')
    if header:
        file.write(' '.join(f"{str(x):>20}" for x in args) + '\n')
    else:
        file.write(' '.join(f"{x:20.15g}" if isinstance(x, (int, float)) else str(x) for x in args) + '\n')
    file.close()

    

if __name__ == '__main__':

    # Method Bisection
    dis = [0.0, 10.0]
    thresh = 1e-10
    bisection(dis, thresh)

    # # Method Fixed Point
    # thresh = 1e-100
    # p = [0, 1]
    # N_max = 100
    # fixed_point(p, thresh, N_max)

    # # Method Newton-Rapson
    # thresh = 1e-8
    # p = [0, 1]
    # N_max = 100 
    # newton_rapson(p, thresh, N_max)

    # # Method Secant
    # thresh = 1e-8
    # p = [0, 1+1e-5, 1]
    # N_max = 100 
    # secant(p, thresh, N_max)