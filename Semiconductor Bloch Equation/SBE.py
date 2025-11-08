import numpy as np
import matplotlib.pyplot as plt

# --- 1. Hằng số và Tham số ---
N = 100                     # Số điểm năng lượng 
EPSILON_MAX = 300.0         # meV
DELTA_EPSILON = EPSILON_MAX / N
E_R = 4.2                   # meV
HBAR = 658.5                # meV fs
DELTA_T_LASER = 25.0        # fs (độ rộng xung laser)
CHI_0 = 0.1                 # Cường độ xung (chọn 1.0 từ 0.1-2)
DELTA_0 = 30.0              # meV (năng lượng trội)
T_2 = 210.0                 # fs (thời gian khử pha)

# Tham số mô phỏng
T_0 = -3 * DELTA_T_LASER    # Thời gian bắt đầu 
T_MAX = 500.0               # fs
DT = 2.0                    # fs (bước thời gian)

# --- 2. Hàm trợ giúp g(n, n1) ---
def g_func(n_idx, n1_idx, delta_eps):
    """
    Tính hàm g(n, n1).
    Sử dụng chỉ số mảng (0-based) và chuyển sang 1-based (n, n1) bên trong.
    """
    n = n_idx + 1
    n1 = n1_idx + 1

    if n == n1:
        # Xử lý trường hợp kỳ dị n=n1, ln|.../(n-n1)| -> vô cực.
        # Trong vật lý, tương tác của một hạt với chính nó
        # thường được loại bỏ hoặc xử lý riêng. Ở đây ta đặt bằng 0.
        return 0.0

    term1 = 1.0 / np.sqrt(n * delta_eps)
    term2 = np.log(np.abs((np.sqrt(n) + np.sqrt(n1)) / (np.sqrt(n) - np.sqrt(n1))))
    return term1 * term2

# --- 3. Tiền xử lý ma trận g ---
# Tính toán trước ma trận g_matrix (N x N) để tăng tốc độ
print("Đang tính toán ma trận g(n, n1)...")
g_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        g_matrix[i, j] = g_func(i, j, DELTA_EPSILON)
print("Đã xong ma trận g.")

# Vector hóa các chỉ số n (1-based) và mảng năng lượng
n_vector_1_based = np.arange(1, N + 1)
epsilon_n_array = n_vector_1_based * DELTA_EPSILON
# Trọng số cho mật độ trạng thái (DOS)
dos_weights = np.sqrt(epsilon_n_array)

# --- 4. Các hàm Vector hóa (E_n và Omega_R)  ---

def E_vector(Y, g_mat, delta_eps, e_r):
    """
    Tính vector E_n cho tất cả n.
    Y là mảng trạng thái [2, N].
    """
    f_e = Y[0, :].real  # Phần thực của Y[0,n] là f_e,n
    f_h = Y[0, :].imag  # Phần ảo của Y[0,n] là f_h,n
    f_sum = f_e + f_h
    
    # Phép nhân Matrix-Vector: (N x N) @ (N,) -> (N,)
    sum_term_vector = g_mat @ f_sum
    
    prefactor = (np.sqrt(e_r) / np.pi) * delta_eps
    return prefactor * sum_term_vector

def Omega_R_vector(t, Y, g_mat, delta_eps, e_r, hbar, delta_t, chi_0):
    """
    Tính vector Omega_n^R cho tất cả n.
    """
    p_n_vector = Y[1, :] # Y[1,n] là p_n (số phức)
    
    # --- Xung Laser  ---
    exponent = -(t**2) / (delta_t**2)
    laser_term_scalar = 0.5 * (hbar * np.sqrt(np.pi) / delta_t) * chi_0 * np.exp(exponent)
    
    # --- Tương tác Coulomb  ---
    sum_term_vector = (np.sqrt(e_r) / np.pi) * delta_eps * (g_mat @ p_n_vector)
    
    return (1.0 / hbar) * (laser_term_scalar + sum_term_vector)

# --- 5. Hàm vi phân F(t, Y)  ---
def F(t, Y, g_mat):
    """
    Tính đạo hàm dY/dt = F(t, Y).
    Y là mảng [2, N] (complex)
    Trả về dYdt cũng là mảng [2, N] (complex)
    """
    dYdt = np.zeros_like(Y, dtype=complex)
    
    # Lấy mảng Y_1 (chứa f_e, f_h) và Y_2 (chứa p_n)
    Y_1 = Y[0, :]
    Y_2 = Y[1, :]
    
    # 1. Tính các vector trợ giúp
    E_vec = E_vector(Y, g_mat, DELTA_EPSILON, E_R)
    Omega_R_vec = Omega_R_vector(t, Y, g_mat, DELTA_EPSILON, E_R, HBAR, DELTA_T_LASER, CHI_0)
    
    # 2. Tính dY[0, n]/dt (đạo hàm của f_e + i*f_h) 
    # a = -2 * Im[Omega_R * p_n*]
    a_vec = -2.0 * np.imag(Omega_R_vec * np.conjugate(Y_2))
    # F[0,n] = complex(a, a) -> d(f_e)/dt = a và d(f_h)/dt = a
    dYdt[0, :] = a_vec + 1j * a_vec
    
    # 3. Tính dY[1, n]/dt (đạo hàm của p_n) 
    term1 = (-1j / HBAR) * (epsilon_n_array - DELTA_0 - E_vec) * Y_2
    term2 = 1j * (1.0 - Y_1.real - Y_1.imag) * Omega_R_vec
    term3 = -Y_2 / T_2
    
    dYdt[1, :] = term1 + term2 + term3
    
    return dYdt

# --- 6. Bộ giải RK4 ---
def rk4_step(t, Y, dt, g_mat):
    """Một bước Runge-Kutta bậc 4."""
    k1 = F(t, Y, g_mat)
    k2 = F(t + 0.5 * dt, Y + 0.5 * dt * k1, g_mat)
    k3 = F(t + 0.5 * dt, Y + 0.5 * dt * k2, g_mat)
    k4 = F(t + dt, Y + dt * k3, g_mat)
    
    Y_next = Y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Y_next

# --- 7. Vòng lặp Mô phỏng chính ---
print("Bắt đầu mô phỏng...")
time_points = np.arange(T_0, T_MAX + DT, DT)
print(time_points)
n_steps = len(time_points)
results_f_e_n = []
results_f_h_n = []
results_p_n = []
# Khởi tạo mảng trạng thái Y 
# Y[0,:] = f_e + i*f_h
# Y[1,:] = p_n
Y_current = np.zeros((2, N), dtype=complex)

# Lưu kết quả
results_N_t = [] # Mật độ toàn phần N(t)
results_P_t_complex = [] # Độ phân cực phức P(t)
results_P_t = [] # Độ lớn phân cực |P(t)|
results_f_e_final = None # Hàm phân bố cuối cùng

for i, t in enumerate(time_points):
    if i % (n_steps // 10) == 0:
        print(f"Tiến độ: {i / n_steps * 100:.0f}% (t = {t:.2f} fs)")
        
    # Tính các đại lượng cần quan tâm TẠI thời điểm t (TRƯỚC khi bước)
    # Dùng trọng số DOS (sqrt(epsilon) ~ sqrt(n))
    f_e_n = Y_current[0, :].real
    f_h_n = Y_current[0, :].imag
    p_n = Y_current[1, :]
    
    # N(t) = 2 * sum(f_e,k) -> C * sum(sqrt(n) * f_e,n)
    # (Bỏ qua hằng số C0)
    N_t = np.sum(f_e_n)
    
    # P(t) = sum(p_k) -> C' * sum(sqrt(n) * p_n)
    P_t_complex = np.sum(p_n)
    results_f_e_n.append(f_e_n)
    results_f_h_n.append(f_h_n)
    results_p_n.append(p_n)
    results_N_t.append(N_t)
    results_P_t_complex.append(P_t_complex)
    results_P_t.append(np.abs(P_t_complex))

    # Thực hiện bước RK4 để tính Y tại t+dt
    Y_current = rk4_step(t, Y_current, DT, g_matrix)
print("Mô phỏng hoàn tất.")

# --- 8. Biến đổi Fourier cho P(t) và E(t) ---
def calculate_ft_energy(time_array, signal_array, energy_array, hbar_val):
    """
    Vectorized FT over an energy grid (energy_array in same units as HBAR, e.g. meV).
    Uses convention: FT(E) = ∫ dt S(t) * exp(-i * E * t / ħ)
    Returns complex array of length len(energy_array).
    """
    t = time_array
    # shape (T,)
    dt = t[1] - t[0]
    # build exponent matrix: shape (len(E), len(t))
    # avoid huge memory by chunking if needed; for moderate sizes it's OK
    phase = np.exp(-1j * np.outer(energy_array, t) / hbar_val)  # shape (E, T)
    # trapezoidal integration along time axis:
    ft = np.trapz(signal_array[None, :] * phase, t, axis=1)
    return ft

# --- Usage ---
energy_array = np.linspace(0.0, 500.0, 10000)  # meV
P_omega = calculate_ft_energy(time_points, np.array(results_P_t_complex), energy_array, HBAR)
E_t_array = np.exp(-(time_points**2) / (DELTA_T_LASER**2))
E_omega = calculate_ft_energy(time_points, E_t_array, energy_array, HBAR)

# guard against tiny denominators
eps = 1e-16
alpha_omega = np.imag(P_omega / (E_omega + eps))

# --- 8. Vẽ đồ thị kết quả  ---
plt.figure(figsize=(12, 10))

# Đồ thị 1: Mật độ toàn phần N(t)
plt.subplot(3, 1, 1)
plt.plot(time_points, results_N_t)
plt.title("Mật độ electron toàn phần $N(t)$ (theo trọng số $\sqrt{\epsilon}$)")
plt.xlabel("Thời gian $t$ (fs)")
plt.ylabel("$N(t)$")
plt.grid(True)

# Đồ thị 2: Độ lớn phân cực |P(t)|
plt.subplot(3, 1, 2)
plt.plot(time_points, results_P_t)
plt.title("Độ lớn phân cực toàn phần $|P(t)|$ (theo trọng số $\sqrt{\epsilon}$)")
plt.xlabel("Thời gian $t$ (fs)")
plt.ylabel("$|P(t)|$")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(energy_array, alpha_omega)
plt.title("Phổ hấp thụ")
plt.xlabel("Năng lượng $\epsilon$ (meV)")
plt.ylabel("Hệ số hấp thụ $\\alpha(\epsilon)$")
plt.grid(True)

# # Đồ thị 3: Hàm phân bố cuối cùng f_e(epsilon)
# plt.subplot(3, 1, 3)
# plt.plot(epsilon_n_array, results_f_e_final, 'o-')
# plt.title(f"Hàm phân bố $f_e(\epsilon, t={T_MAX} fs)$")
# plt.xlabel("Năng lượng $\epsilon$ (meV)")
# plt.ylabel("$f_e$")
# plt.grid(True)
# plt.tight_layout()

with open("sbe_simulation_results3.txt", "w") as f:
    for i in range(time_points.shape[0]):
        f.write(f"{time_points[i]}\t{results_N_t[i]:.6f}\t{results_P_t[i]:.6f}\n")

with open("sbe_f_e_n_results.txt", "w") as f:
    for i in range(time_points.shape[0]):
        # f.write(f"{time_points[i]}\t")
        for n in range(N):
            f.write(f"{results_f_e_n[i][n]:.6f}\t")
        f.write("\n")

with open("sbe_p_n_results.txt", "w") as f:
    for i in range(time_points.shape[0]):
        # f.write(f"{time_points[i]}\t")
        for n in range(N):
            f.write(f"{results_p_n[i][n]:.6f}\t")
        f.write("\n")

# with open("sbe_absorption_results.txt", "w") as f:
#     # alpha_omega is defined on the energy axis (epsilon_n_array), length N.
#     # Write energy (meV) and absorption alpha for each energy sample.
#     for i in range(len(omega_array)):
#         f.write(f"{omega_array[i]:.6f}\t{alpha_omega[i]:.6f}\n")

# Lưu đồ thị
plt.tight_layout()
plt.savefig("sbe_simulation_results.png")
print("Đã lưu kết quả đồ thị vào file 'sbe_simulation_results.png'")