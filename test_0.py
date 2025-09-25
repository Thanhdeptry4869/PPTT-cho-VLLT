import numpy as np

def bisection(dis, thresh, g_num):
    
    diff = thresh + 1
    N = 0
    export_file(f'Bisection Method for function {g_num}','\n','-'*140,'\n','-'*140,'\n','N', 'a', 'b', 'ref', 'f(ref)','\n', '-'*140, header=True)
    while diff > thresh:
        N += 1
        ref = (dis[0] + dis[1])/2

        export_file(N, dis[0], dis[1], ref, func(ref)[g_num])

        if func(ref)[g_num] == 0:
            print(f'The result is {ref}')
            break
        elif func(dis[0])[g_num] * func(ref)[g_num] < 0:
            dis[1] = ref
        elif func(ref)[g_num] * func(dis[1])[g_num] < 0:
            dis[0] = ref
        else:
            print('-'*30)
            print(f'This distance does not have a root, after {N} iterations')
            print('-'*30)
            raise ValueError
        
        diff = np.abs(dis[0] - dis[1])
        if np.abs(dis[0] - dis[1]) < thresh:
            print('The result is {ans} after {N} iterations'.format(ans = (dis[0] + dis[1])/2, N = N))
            #print('diff  = ', diff)
    export_file('-'*140)

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

def newton_rapson(p, thresh, N_max):
    diff = thresh + 1
    N = 0
    while diff > thresh:
        p[0] = p[1] - func(p[1])[2] / func(p[1])[3]
        diff = np.abs(p[0] - p[1])
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

def func(x, method = True):

    ##########################################################
    if method:
        g0_bisection = x - 2**(-x)
        g1_bisection = np.exp(x) - 2 - np.cos(np.exp(x) - 2)
        return g0_bisection, g1_bisection
    ##########################################################
    else:
        g0_fixedpoint= np.sqrt(10 - x**3)/2
        g1_fixedpoint= x - (x**3 + 4*x**2 - 10)
        g2_fixedpoint= np.sqrt(10/x - 4*x)
        g3_fixedpoint= np.sqrt(10/(x + 4))
        g4_fixedpoint= x - (x**3 + 4*x**2 - 10)/(3*x**2 + 8*x)
        return g0_fixedpoint, g1_fixedpoint, g2_fixedpoint, g3_fixedpoint, g4_fixedpoint

def export_file(*args, header=False):
    file = open('Alte_Week/BT_4/results.dat', 'a')
    if header:
        file.write(' '.join(f"{str(x):>20}" for x in args) + '\n')
    else:
        file.write(' |'.join(f"{x:20.15g}" if isinstance(x, (int, float)) else str(x) for x in args) + '\n')
    file.close()

# # Method Bisection
# g_num = 1
# dis = [0.5, 1.5]
# thresh = 1e-10
# bisection(dis, thresh, g_num)

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