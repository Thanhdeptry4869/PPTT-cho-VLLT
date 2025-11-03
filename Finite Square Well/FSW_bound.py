import numpy as np
import matplotlib.pyplot as plt

class methods:
    @staticmethod
    def bisection(func, a, b, N=1000, epsilon=1e-10):
        a_vals = []
        b_vals = []
        c_vals = []
        f_vals = []
        if func(a) * func(b) >= 0:
            raise ValueError("Choose f(a) * f(b) < 0 please!")
        else:
            for i in range(N):
                c = (a + b) / 2
                fc = func(c)
                a_vals.append(a)
                b_vals.append(b)
                c_vals.append(c)
                f_vals.append(fc)
                if abs(fc) <= epsilon:
                    print(f"Converged after {i} steps, root = {c}")
                    return c, a_vals, b_vals, c_vals, f_vals
                elif fc * func(a) < 0:
                    b = c
                elif fc * func(b) < 0:
                    a = c
            raise ValueError("Can't find solution in given max_iter")

    @staticmethod
    def newton_rapson(g, g_prime, p0, N=1000, epsilon=1e-10):
        g_vals = []
        for i in range(N):
            try:
                p = p0 - g(p0) / g_prime(p0)            # may overflow or return weird types
                p = float(p)         # enforce float, triggers OverflowError early
                if not np.isfinite(p):   # catches NaN or Inf
                    print(f"Diverged (NaN/Inf) at iteration {i}")
                    return None, g_vals
            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                return None, g_vals

            g_vals.append(p)
            if abs(p - p0) < epsilon:
                print(f"Converged after {i} steps, p = {p}")
                return p, g_vals
            p0 = p
        print("Exceeded maximum iteration or diverged/undefined")
        return p, g_vals

    @staticmethod
    def secant(func, p0, p1, N=1000, epsilon=1e-10):
        f_vals = []
        p_vals = [p0, p1]
        
        for i in range(N):
            try:
                p = p1 - func(p1) * (p1 - p0) / (func(p1) - func(p0))  # Secant formula
                p = float(p)  # enforce float, triggers OverflowError early
                if not np.isfinite(p):  # catches NaN or Inf
                    # print(f"Diverged (NaN/Inf) at iteration {i}")
                    return None, p_vals, f_vals
            except Exception as e:
                # print(f"Error at iteration {i}: {e}")
                return None, p_vals, f_vals

            f_vals.append(func(p))
            p_vals.append(p)
            
            if abs(p - p1) < epsilon:
                # print(f"Converged after {i} steps, p = {p}")
                return p, p_vals, f_vals
            
            p0, p1 = p1, p
        
        print("Exceeded maximum iteration or diverged/undefined")
        return p, p_vals, f_vals
    
class Solver(methods):
    def __init__(self, model, method_name='bisection'):
        self.model = model
        self.method_name = method_name

    def get_roots(self, parity='even', N=1000, epsilon=1e-6):
        func = self.model.z_even if parity == 'even' else self.model.z_odd
        derivative = self.model.z_even_derivative if parity == 'even' else self.model.z_odd_derivative
        z0 = self.model.z0

        # Chia các khoảng giữa các tiệm cận tan(z)
        intervals = [(i*np.pi/2 + 1e-3, (i+1)*np.pi/2 - 1e-3) for i in range(int(z0 // (np.pi/2)) + 1)]
        roots = []

        for a, b in intervals:
            try:
                if self.method_name == 'bisection':
                    root, *_ = self.bisection(func, a, b, N, epsilon)
                elif self.method_name == 'secant':
                    root, *_ = self.secant(func, a+1e-3, b-1e-3, N, epsilon)
                elif self.method_name == 'newton':
                    root, _ = self.newton_rapson(func, derivative, (a+b)/2, N, epsilon)
                else:
                    raise ValueError(f"Unknown method {self.method_name}")
                
                if root is not None and a < root < b:
                    roots.append(root)
            except ValueError:
                continue
            
            roots = list(set(round(x, 4) for x in roots))  # Loại bỏ nghiệm trùng lặp và làm tròn
        return roots

class FSW_bound:
    def __init__(self, V0, a, m, hbar):
        self.V0 = V0
        self.a = a
        self.m = m
        self.hbar = hbar
        self.z0 = a * np.sqrt(2 * m * V0) / hbar
    
    def z_even(self, z):
        return np.tan(z) - np.sqrt(self.z0**2 - z**2) / z
    
    def z_odd(self, z):
        return np.tan(z) + z / np.sqrt(self.z0**2 - z**2)
    
    def z_even_derivative(self, z):
        return 1 / (np.cos(z)**2) + self.z0**2 / (z**2 * np.sqrt(self.z0**2 - z**2))

    def z_odd_derivative(self, z):
        return 1 / (np.cos(z)**2) - self.z0**2 / (self.z0**2 - z**2)**1.5

    def z_to_energy(self, z):
        return (self.hbar**2 * z**2) / (2 * self.m * self.a**2) - self.V0

def get_wavefunction(energy, model, parity='even', num_points=1000):
    # k là số sóng bên TRONG giếng
    k = np.sqrt(2 * model.m * (model.V0 + energy)) / model.hbar
    # kappa là hệ số tắt dần bên NGOÀI giếng
    kappa = np.sqrt(-2 * model.m * energy) / model.hbar
    
    x_inside = np.linspace(-model.a, model.a, num_points)
    x_outside_left = np.linspace(-3*model.a, -model.a, num_points)
    x_outside_right = np.linspace(model.a, 3*model.a, num_points)
    
    if parity == 'even':
        psi_inside = np.cos(k * x_inside)
        # Hệ số để đảm bảo tính liên tục tại x=a
        coeff = np.cos(k * model.a) / np.exp(-kappa * model.a)
    else: # parity == 'odd'
        psi_inside = np.sin(k * x_inside)
        # Hệ số để đảm bảo tính liên tục tại x=a
        coeff = np.sin(k * model.a) / np.exp(-kappa * model.a)
    
    # Hàm sóng bên ngoài phải tắt dần
    psi_outside_left = coeff * np.exp(kappa * x_outside_left) # Chú ý dấu +
    psi_outside_right = -coeff * np.exp(-kappa * x_outside_right) if parity == 'odd' and x_outside_left[0] < 0 else coeff * np.exp(-kappa * x_outside_right)

    # Chỉnh sửa để hàm lẻ đối xứng đúng
    if parity == 'odd':
        psi_outside_left = -coeff * np.exp(kappa * x_outside_left)
        psi_outside_right = coeff * np.exp(-kappa * x_outside_right)
    else: # parity == 'even'
        psi_outside_left = coeff * np.exp(kappa * x_outside_left)
        psi_outside_right = coeff * np.exp(-kappa * x_outside_right)

    x = np.concatenate((x_outside_left, x_inside, x_outside_right))
    psi = np.concatenate((psi_outside_left, psi_inside, psi_outside_right))
    return x, psi

if __name__ == "__main__":
    # Ví dụ sử dụng
    model = FSW_bound(V0=10, a=1, m=1, hbar=1)
    solver = Solver(model, method_name='bisection')
    
    even_roots = solver.get_roots(parity='even')
    odd_roots = solver.get_roots(parity='odd')
    
    print("Even roots (z):", even_roots)
    print("Odd roots (z):", odd_roots)
    
    for root in even_roots:
        energy = model.z_to_energy(root)
        x, psi = get_wavefunction(energy, model, parity='even')
        plt.plot(x, psi, label=f'Even E={energy:.4f}')
    
    for root in odd_roots:
        energy = model.z_to_energy(root)
        x, psi = get_wavefunction(energy, model, parity='odd')
        plt.plot(x, psi, label=f'Odd E={energy:.4f}')
    
    plt.title('Wavefunctions for Bound States in Finite Square Well')
    plt.xlabel('x')
    plt.ylabel('ψ(x)')
    plt.axvline(x=-model.a, color='k', linestyle='--')
    plt.axvline(x=model.a, color='k', linestyle='--')
    plt.legend()
    plt.show()