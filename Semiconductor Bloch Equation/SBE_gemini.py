import numpy as np
import matplotlib.pyplot as plt

# --- 1. Hằng số và Tham số ---
N = 100                     # Số điểm năng lượng
EPSILON_MAX = 300.0         # meV
DELTA_EPSILON = EPSILON_MAX / N
E_R = 4.2                   # meV
HBAR = 658.5                # meV fs
DELTA_T_LASER = 25.0        # fs (độ rộng xung laser)
CHI_0 = 0.1                 # Cường độ xung
DELTA_0 = 30.0              # meV (năng lượng trội)
T_2 = 210.0                 # fs (thời gian khử pha)

# Tham số mô phỏng
T_0 = -3 * DELTA_T_LASER    # Thời gian bắt đầu
T_MAX = 1500.0              # fs (Đã tăng để P(t) tắt hẳn)
DT = 2.0                    # fs (bước thời gian)

# --- 2. Hàm trợ giúp g(n, n1) ---
def g_func(n_idx, n1_idx, delta_eps):
    n = n_idx + 1
    n1 = n1_idx + 1
    sn, sn1 = np.sqrt(n), np.sqrt(n1)
    if abs(sn - sn1) < 1e-12:
        # Xử lý kỳ dị (Giống code mới của bạn)
        return 2.0 / np.sqrt(n * delta_eps)
    term1 = 1.0 / np.sqrt(n * delta_eps)
    term2 = np.log(np.abs((sn + sn1) / (sn - sn1)))
    return term1 * term2

# --- 3. Tiền xử lý ma trận g ---
print("Đang tính toán ma trận g(n, n1)...")
g_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        g_matrix[i, j] = g_func(i, j, DELTA_EPSILON)
print("Đã xong ma trận g.")

# Vector hóa các chỉ số n (1-based) và mảng năng lượng
n_vector_1_based = np.arange(1, N + 1)
epsilon_n_array = n_vector_1_based * DELTA_EPSILON
# *** LỖI VẬT LÝ 2 ĐÃ SỬA: Trọng số DOS theo PDF là sqrt(n) ***
dos_weights = np.sqrt(n_vector_1_based)

# --- 4. Các hàm Vector hóa (E_n và Omega_R) ---
def E_vector(Y, g_mat, delta_eps, e_r):
    f_e = Y[0, :].real
    f_h = Y[0, :].imag
    f_sum = f_e + f_h
    sum_term_vector = g_mat @ f_sum
    prefactor = (np.sqrt(e_r) / np.pi) * delta_eps
    return prefactor * sum_term_vector

def Omega_R_vector(t, Y, g_mat, delta_eps, e_r, hbar, delta_t, chi_0):
    p_n_vector = Y[1, :]
    
    # --- Xung Laser ---
    exponent = -(t**2) / (delta_t**2)
    # *** LỖI VẬT LÝ 3 ĐÃ SỬA: Hệ số 0.25 (1/2 * 1/2) theo file PDF ***
    laser_term_scalar = 0.5 * (hbar * np.sqrt(np.pi) / delta_t) * chi_0 * np.exp(exponent)
    
    # --- Tương tác Coulomb ---
    sum_term_vector = (np.sqrt(e_r) / np.pi) * delta_eps * (g_mat @ p_n_vector)
    
    return (1.0 / hbar) * (laser_term_scalar + sum_term_vector)

# --- 5. Hàm vi phân F(t, Y) ---
def F(t, Y, g_mat):
    dYdt = np.zeros_like(Y, dtype=complex)
    Y_1 = Y[0, :]
    Y_2 = Y[1, :]
    
    E_vec = E_vector(Y, g_mat, DELTA_EPSILON, E_R)
    Omega_R_vec = Omega_R_vector(t, Y, g_mat, DELTA_EPSILON, E_R, HBAR, DELTA_T_LASER, CHI_0)
    
    a_vec = -2.0 * np.imag(Omega_R_vec * np.conjugate(Y_2))
    dYdt[0, :] = a_vec + 1j * a_vec
    
    term1 = (-1j / HBAR) * (epsilon_n_array - DELTA_0 - E_vec) * Y_2
    term2 = 1j * (1.0 - Y_1.real - Y_1.imag) * Omega_R_vec
    term3 = -Y_2 / T_2
    
    dYdt[1, :] = term1 + term2 + term3
    
    return dYdt

# --- 6. Bộ giải RK4 ---
def rk4_step(t, Y, dt, g_mat):
    k1 = F(t, Y, g_mat)
    k2 = F(t + 0.5 * dt, Y + 0.5 * dt * k1, g_mat)
    k3 = F(t + 0.5 * dt, Y + 0.5 * dt * k2, g_mat)
    k4 = F(t + dt, Y + dt * k3, g_mat)
    Y_next = Y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Y_next

# --- 7. Vòng lặp Mô phỏng chính ---
print("Bắt đầu mô phỏng...")
time_points = np.arange(T_0, T_MAX + DT, DT)
n_steps = len(time_points)
Y_current = np.zeros((2, N), dtype=complex)

results_f_e_n = []  # Lưu f_e(n) theo thời gian
results_p_n = []    # Lưu p_n theo thời gian
results_N_t = []
results_P_t_complex = [] # Mảng complex P(t)
results_P_t = []

for i, t in enumerate(time_points):
    if i % (n_steps // 10) == 0:
        print(f"Tiến độ: {i / n_steps * 100:.0f}% (t = {t:.2f} fs)")
        
    f_e_n = Y_current[0, :].real
    p_n = Y_current[1, :]
    
    N_t = 2.0 * np.sum(f_e_n * dos_weights)
    P_t_complex = np.sum(p_n * dos_weights)
    
    results_N_t.append(N_t)
    results_P_t_complex.append(P_t_complex)
    results_P_t.append(np.abs(P_t_complex))
    
    # Lưu trạng thái phân bố và phân cực tại thời điểm này
    results_f_e_n.append(f_e_n.copy())
    results_p_n.append(p_n.copy())

    Y_current = rk4_step(t, Y_current, DT, g_matrix)
print("Mô phỏng hoàn tất.")

# --- 8. Biến đổi Fourier cho P(t) và E(t) ---
def calculate_ft_energy(time_array, signal_array, energy_axis, hbar_val):
    """
    Tính FT bằng Tổng Riemann.
    Quy ước FT(E) = ∫ dt S(t) * exp(+i * E * t / ħ) (Theo PDF)
    """
    t = time_array
    dt = t[1] - t[0]
    
    # *** LỖI TÍNH TOÁN 1 ĐÃ SỬA: Dùng dấu CỘNG (+1j) theo file PDF ***
    phase = np.exp(1j * np.outer(energy_axis, t) / hbar_val)  # shape (E, T)
    
    # Dùng tổng Riemann đơn giản
    ft = np.sum(signal_array * phase, axis=1) * dt
    return ft

# --- Usage ---
# *** LỖI TÍNH TOÁN 2 ĐÃ SỬA: Dùng trục tuyến tính, không dùng fftfreq ***
print("Đang tính toán Phổ Hấp thụ...")
energy_axis_plot = np.linspace(-300, 300, 1000) # Trục năng lượng (meV) tuyến tính

P_omega = calculate_ft_energy(time_points, np.array(results_P_t_complex), energy_axis_plot, HBAR)
E_t_array = np.exp(-(time_points**2) / (DELTA_T_LASER**2))
E_omega = calculate_ft_energy(time_points, E_t_array, energy_axis_plot, HBAR)

# Tính phổ hấp thụ
eps = 1e-16
alpha_omega = np.imag(P_omega / (E_omega + eps))

# Chuẩn hóa (tùy chọn, để dễ xem)
alpha_omega = alpha_omega / np.max(np.abs(alpha_omega))

# --- 9. Vẽ đồ thị kết quả ---
plt.figure(figsize=(12, 10))

# Đồ thị 1: Mật độ toàn phần N(t)
plt.subplot(3, 1, 1)
plt.plot(time_points, results_N_t)
plt.title("Mật độ electron toàn phần $N(t)$ (Đã sửa DOS/Spin)")
plt.xlabel("Thời gian $t$ (fs)")
plt.ylabel("$N(t)$")
plt.grid(True)

# Đồ thị 2: Độ lớn phân cực |P(t)|
plt.subplot(3, 1, 2)
plt.plot(time_points, results_P_t)
plt.title("Độ lớn phân cực toàn phần $|P(t)|$ (Đã sửa DOS)")
plt.xlabel("Thời gian $t$ (fs)")
plt.ylabel("$|P(t)|$")
plt.grid(True)

# Đồ thị 3: Phổ hấp thụ
plt.subplot(3, 1, 3)
# *** ĐÃ SỬA: Dùng trục tuyến tính (energy_axis_plot) ***
plt.plot(energy_axis_plot, alpha_omega)
plt.title("Phổ hấp thụ (Đã sửa lỗi FT)")
plt.xlabel("Năng lượng Photon (meV)")
plt.ylabel("Hệ số hấp thụ (Chuẩn hóa)")
# Thêm vạch chỉ 30 meV để kiểm tra
plt.axvline(x=30.0, color='r', linestyle='--', label='$\\Delta_0 = 30$ meV')
plt.xlim(-100, 100)
plt.legend()
plt.grid(True)

with open("sbe_simulation_results3.txt", "w") as f:
    for i in range(len(time_points)):
        f.write(f"{time_points[i]}\t{results_N_t[i]:.6f}\t{results_P_t[i]:.6f}\n")

with open("sbe_f_e_n_results.txt", "w") as f:
    for i in range(len(time_points)):
        # f.write(f"{time_points[i]}\t")
        for n in range(N):
            f.write(f"{results_f_e_n[i][n]:.6f}\t")
        f.write("\n")

with open("sbe_p_n_results.txt", "w") as f:
    for i in range(len(time_points)):
        # f.write(f"{time_points[i]}\t")
        for n in range(N):
            # p_n is complex so write real+imag form to avoid formatting errors
            val = results_p_n[i][n]
            f.write(f"{val.real:.6f}+{val.imag:.6f}j\t")
        f.write("\n")

# Save absorption spectrum results
with open("sbe_absorption_results.txt", "w") as f:
    for i in range(len(energy_axis_plot)):
        f.write(f"{energy_axis_plot[i]:.6f}\t{alpha_omega[i]:.6f}\n")


plt.tight_layout()
plt.savefig("sbe_simulation_results_FIXED_FT.png")
print("Đã lưu kết quả đồ thị vào file 'sbe_simulation_results_FIXED_FT.png'")