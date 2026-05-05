import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
import cvxpy as cp

# ==================================================
# LÊ ARQUIVO TXT NO FORMATO:
# x,y
# 10.5,20.3
# 15.0,22.1
# ...
# valores em milímetros
# ==================================================
def load_key_from_txt(filename):
    key = np.loadtxt(filename, delimiter=",")
    return key

def build_centerline_from_key(key_xy, n_points=600):
    x_key = key_xy[:, 0]
    y_key = key_xy[:, 1]

    # spline periódica (pista fechada)
    tck, _ = splprep([x_key, y_key], s=0.0, per=1)

    # amostragem uniforme no parâmetro
    u_fine = np.linspace(0.0, 1.0, n_points, endpoint=False)
    x_ref, y_ref = splev(u_fine, tck)

    return np.array(x_ref), np.array(y_ref)

def compute_normals(x_ref, y_ref):
    # derivadas (tangente aproximada)
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)

    t_norm = np.sqrt(dx**2 + dy**2) + 1e-12
    tx = dx / t_norm
    ty = dy / t_norm

    # normal = rotação de 90° da tangente (para a direita)
    nx = ty
    ny = -tx

    n_norm = np.sqrt(nx**2 + ny**2) + 1e-12
    nx /= n_norm
    ny /= n_norm

    return nx, ny

def compute_curvature(x_ref, y_ref):
    """
    Curvatura contínua aproximada da linha central.
    """
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    ds = np.sqrt(dx**2 + dy**2) + 1e-12

    # fórmula da curvatura plana: kappa = (x' y'' - y' x'') / |r'|^3
    kappa = (dx * ddy - dy * ddx) / (ds**3 + 1e-12)
    return kappa

def compute_discrete_second_differences(x, y):
    """
    Segundas diferenças discretas de x e y:
    dx2[i] = x[i+2] - 2*x[i+1] + x[i]
    idem para y.
    """
    dx2 = x[2:] - 2 * x[1:-1] + x[:-2]
    dy2 = y[2:] - 2 * y[1:-1] + y[:-2]
    return dx2, dy2

def optimize_weighted_min_curvature_with_local_bounds(
    x_ref, y_ref, nx, ny, half_width,
    gamma_curv=50.0,
    power_curv=2.0,
    lambda_alpha=1e-3,
    disc_rel_thresh=0.7,
    curv_factor=1.0,
    kappa_side_thresh=1e-5
):
    """
    Otimiza curvatura com pesos locais e limites em regiões mais fechadas.
    """
    N = len(x_ref)

    # variável de deslocamento lateral
    alpha = cp.Variable(N)

    # trajetória otimizada
    x = x_ref + cp.multiply(alpha, nx)
    y = y_ref + cp.multiply(alpha, ny)

    # segundas diferenças (aprox. segunda derivada)
    dx2 = x[2:] - 2 * x[1:-1] + x[:-2]
    dy2 = y[2:] - 2 * y[1:-1] + y[:-2]

    # --- Curvatura contínua da referência ---
    kappa_ref = compute_curvature(x_ref, y_ref)  # tamanho N
    kappa_mid = np.abs(kappa_ref[1:-1])          # alinhar com dx2/dy2 (N-2)

    # --- Peso de curvatura baseado em kappa_ref ---
    max_k = np.max(kappa_mid) + 1e-12
    k_norm = (kappa_mid / max_k) ** power_curv   # em [0,1]
    weights = 1.0 + gamma_curv * k_norm          # >=1, maior em hairpins

    # termo de curvatura ponderado:
    curv_term = cp.sum(
        cp.multiply(weights, cp.square(dx2)) +
        cp.multiply(weights, cp.square(dy2))
    )

    # regularização leve em alpha (não grudar 100% na borda)
    alpha_term = lambda_alpha * cp.sum_squares(alpha)

    objective = cp.Minimize(curv_term + alpha_term)

    # restrições básicas
    constraints = [
        cp.abs(alpha) <= half_width,
        alpha[0] == alpha[-1],  # pista fechada
    ]

    # --- Limite local de curvatura discreta nos pontos mais fechados ---
    dx2_ref, dy2_ref = compute_discrete_second_differences(x_ref, y_ref)
    curv_disc_ref = np.sqrt(dx2_ref**2 + dy2_ref**2)

    max_cd = np.max(curv_disc_ref) + 1e-12
    thresh_cd = disc_rel_thresh * max_cd

    indices_bound = np.where(curv_disc_ref > thresh_cd)[0]

    for j in indices_bound:
        # j é índice em dx2/dy2 (0..N-3); ponto correspondente em alpha é j+1
        i = j + 1

        # 2a. limite de curvatura nesses pontos
        bound = float(curv_factor * curv_disc_ref[j])
        if bound > 0:
            constraints.append(
                cp.norm(cp.hstack([dx2[j], dy2[j]]), 2) <= bound
            )

        # 3. restrição de lado EXTERNO nesses mesmos pontos (apenas se curva forte)
        if kappa_ref[i] > kappa_side_thresh:
            constraints.append(alpha[i] >= 0)
        elif kappa_ref[i] < -kappa_side_thresh:
            constraints.append(alpha[i] <= 0)
        # se |kappa| pequeno, não força lado

    prob = cp.Problem(objective, constraints)
    # SOC constraints -> usar SCS
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Aviso: otimização falhou, usando linha central.")
        return x_ref, y_ref, np.zeros(N)

    alpha_val = alpha.value
    x_opt = x_ref + alpha_val * nx
    y_opt = y_ref + ny * alpha_val

    return x_opt, y_opt, alpha_val

def get_square_vertices(xc, yc, tx, ty, side=25):
    """
    Gera os vértices de um quadrado com lado 'side' (mm),
    centrado em (xc, yc) e orientado pela tangente (tx, ty).
    """
    t_norm = np.sqrt(tx**2 + ty**2) + 1e-12
    tx /= t_norm
    ty /= t_norm

    nx = ty
    ny = -tx

    h = side / 2.0

    corners = []

    for sx in [-1, 1]:
        for sy in [-1, 1]:
            x_corner = xc + sx * h * tx + sy * h * nx
            y_corner = yc + sx * h * ty + sy * h * ny
            corners.append([x_corner, y_corner])

    return np.array(corners)

def main():

    # ==================================================
    # NOME DO ARQUIVO TXT
    # ==================================================
    arquivo = "coordenadas.txt"

    # lê coordenadas em mm
    key = load_key_from_txt(arquivo)

    # linha central refinada
    x_ref, y_ref = build_centerline_from_key(key, n_points=400)

    # vetores normais à linha central
    nx, ny = compute_normals(x_ref, y_ref)

    # meia largura da pista em mm
    half_width = 200 / 2  # 100 mm = 10 cm

    # bordas da pista
    x_outer = x_ref + half_width * nx
    y_outer = y_ref + half_width * ny

    x_inner = x_ref - half_width * nx
    y_inner = y_ref - half_width * ny

    # trajetória otimizada:
    x_opt, y_opt, alpha = optimize_weighted_min_curvature_with_local_bounds(
        x_ref, y_ref, nx, ny, half_width,
        gamma_curv=50.0,
        power_curv=2.0,
        lambda_alpha=1e-3,
        disc_rel_thresh=0.7,
        curv_factor=1.0,
        kappa_side_thresh=1e-5
    )

    print("Deslocamento lateral médio [mm]:", np.mean(alpha))
    print("Deslocamento lateral máximo módulo [mm]:", np.max(np.abs(alpha)))
    print("Deslocamento máximo para a direita [mm]:", np.max(alpha))
    print("Deslocamento máximo para a esquerda [mm]:", np.min(alpha))

    # ==================================================
    # SALVA TRAJETO OTIMIZADO EM TXT (SEM HEADER)
    # FORMATO: x_mm,y_mm,alpha_mm
    # alpha > 0 direita, alpha < 0 esquerda
    # ==================================================
    coords_opt = np.column_stack((x_opt, y_opt, alpha))
    np.savetxt(
        "trajeto_otimizado.txt",
        coords_opt,
        fmt="%.6f",
        delimiter=","
    )

    # ==================================================
    # PLOT ESTÁTICO
    # ==================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x_ref, y_ref, 'k--', linewidth=0.5, label='Linha central')
    ax.plot(x_outer, y_outer, 'b', linewidth=1.0, label='Borda externa')
    ax.plot(x_inner, y_inner, 'b', linewidth=1.0, label='Borda interna')
    ax.plot(x_opt, y_opt, 'r', linewidth=2.0, label='Trajeto otimizado')

    ax.set_aspect('equal', 'box')
    ax.grid(True)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Pista em milímetros")

    ax.legend()

    # ==================================================
    # ANIMAÇÃO DO "CARRINHO" QUADRADO
    # ==================================================
    dx_opt = np.gradient(x_opt)
    dy_opt = np.gradient(y_opt)

    side = 250  # quadrado 250 mm = 25 cm

    verts0 = get_square_vertices(
        x_opt[0], y_opt[0],
        dx_opt[0], dy_opt[0],
        side=side
    )

    square_patch = Polygon(
        verts0,
        closed=True,
        edgecolor='g',
        facecolor='none',
        linewidth=2
    )

    ax.add_patch(square_patch)

    def update(frame):
        i = frame % len(x_opt)

        verts = get_square_vertices(
            x_opt[i], y_opt[i],
            dx_opt[i], dy_opt[i],
            side=side
        )

        square_patch.set_xy(verts)

        return (square_patch,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_opt),
        interval=20,
        blit=True,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()