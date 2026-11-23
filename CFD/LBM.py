# LBM.py - solver do Projeto 17 (Lattice Boltzmann, modelo D2Q9)
#
# Recebe a mascara do carrinho (True = carrinho, False = ar) e simula o
# vento passando em volta dele. Devolve rho, ux e uy com shape (ny, nx).
#
# Contornos:
#   - esquerda: entrada de vento com velocidade constante u_entrada
#   - direita: saida (copia a coluna do lado, gradiente zero)
#   - topo e fundo: periodicos (o np.roll ja faz isso sozinho)
#   - carrinho: bounce-back
#
# Eixos iguais aos do geometry.py: linha 0 em cima, entao uy > 0 quer
# dizer descendo na imagem (linha aumentando).
#
# Uso:
#   import LBM
#   rho, ux, uy = LBM.simular(mask, u_entrada=0.05, tau=0.6, iteracoes=10000)

import time

import numpy as np

# Direcoes do D2Q9, desenhadas como aparecem na imagem (linha 0 em cima):
#   7 4 8
#   3 0 1
#   6 2 5
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# direcao contraria de cada uma (usada no bounce-back)
oposto = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Ma = u / cs, com cs = 1/sqrt(3). Acima de ~0.3 o LBM perde precisao
U_MAXIMO = 0.3 / np.sqrt(3)


def tau_para_viscosidade(tau):
    return (tau - 0.5) / 3


def viscosidade_para_tau(nu):
    return 0.5 + 3 * nu


def equilibrio(rho, ux, uy):
    # f_eq = w * rho * (1 + 3 e.u + 4.5 (e.u)^2 - 1.5 u^2)
    eu = ex[:, None, None] * ux + ey[:, None, None] * uy
    u2 = ux**2 + uy**2
    return w[:, None, None] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u2)


def macroscopicas(f):
    rho = f.sum(axis=0)
    ux = (f * ex[:, None, None]).sum(axis=0) / rho
    uy = (f * ey[:, None, None]).sum(axis=0) / rho
    return rho, ux, uy


def colisao(f, feq, tau):
    return f - (f - feq) / tau


def streaming(f):
    # cada f_i anda uma celula na sua direcao
    f_novo = np.empty_like(f)
    for i in range(9):
        f_novo[i] = np.roll(f[i], (ey[i], ex[i]), axis=(0, 1))
    return f_novo


def celulas_bounce_back(mask):
    # Para cada direcao i, as celulas de ar que receberiam no streaming algo
    # vindo de dentro do carrinho. Nelas o f_i e substituido pelo que a
    # propria celula mandou na direcao contraria (bounce-back com a parede
    # no meio do caminho entre o ar e o carrinho).
    return [np.roll(mask, (ey[i], ex[i]), axis=(0, 1)) & ~mask for i in range(9)]


def bounce_back(f, f_colisao, celulas):
    for i in range(1, 9):
        c = celulas[i]
        f[i][c] = f_colisao[oposto[i]][c]


def simular(mask, u_entrada=0.05, tau=0.6, iteracoes=10000, viscosidade=None,
            tolerancia=1e-5, intervalo=200, mostrar_progresso=True):
    """
    Roda o LBM e devolve (rho, ux, uy).

    mask:        matriz bool (ny, nx), True = carrinho
    u_entrada:   velocidade do vento em unidades de lattice (0.02 a 0.1 e o ideal)
    tau:         tempo de relaxamento (> 0.5). Se passar a viscosidade, o tau sai dela
    iteracoes:   numero maximo de passos
    tolerancia:  para antes se o escoamento parar de mudar
                 (variacao maxima de ux a cada 'intervalo' passos, relativa a u_entrada).
                 Use 0 para rodar sempre todas as iteracoes.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask tem que ser 2D (ny, nx)")
    if viscosidade is not None:
        tau = viscosidade_para_tau(viscosidade)
    if tau <= 0.5:
        raise ValueError(f"tau tem que ser maior que 0.5 (recebi {tau})")
    if u_entrada <= 0:
        raise ValueError("u_entrada tem que ser positiva (vento da esquerda para a direita)")
    if u_entrada > U_MAXIMO:
        print(f"aviso: u_entrada = {u_entrada} e alta demais pro LBM (Mach > 0.3), "
              f"o ideal e ficar abaixo de {U_MAXIMO:.2f}")
    if tau < 0.51:
        print("aviso: tau muito perto de 0.5, a simulacao pode explodir")
    if mask[:, 0].any() or mask[:, -1].any() or mask[0, :].any() or mask[-1, :].any():
        print("aviso: o carrinho encosta na borda da malha")

    ny, nx = mask.shape

    # comeca com o vento uniforme em todo lugar, parado dentro do carrinho
    rho = np.ones((ny, nx))
    ux = np.full((ny, nx), float(u_entrada))
    uy = np.zeros((ny, nx))
    ux[mask] = 0
    f = equilibrio(rho, ux, uy)

    # f da coluna de entrada (equilibrio com rho = 1 e u = u_entrada)
    f_entrada = equilibrio(np.ones((ny, 1)), np.full((ny, 1), float(u_entrada)), np.zeros((ny, 1)))

    celulas = celulas_bounce_back(mask)
    ux_antes = ux.copy()
    inicio = time.time()

    for it in range(1, iteracoes + 1):
        rho, ux, uy = macroscopicas(f)
        f_colisao = colisao(f, equilibrio(rho, ux, uy), tau)
        f = streaming(f_colisao)
        bounce_back(f, f_colisao, celulas)
        f[:, :, :1] = f_entrada        # entrada
        f[:, :, -1] = f[:, :, -2]      # saida

        if it % intervalo == 0:
            if not np.isfinite(ux).all():
                raise RuntimeError(f"a simulacao divergiu na iteracao {it}: "
                                   "aumente o tau ou diminua o u_entrada")
            variacao = np.abs(ux - ux_antes)[~mask].max() / u_entrada
            ux_antes = ux.copy()
            if mostrar_progresso:
                print(f"iteracao {it:6d} | variacao {variacao:.2e} | {time.time() - inicio:.0f} s")
            if variacao < tolerancia:
                if mostrar_progresso:
                    print(f"convergiu na iteracao {it}")
                break

    rho, ux, uy = macroscopicas(f)
    # dentro do carrinho nao tem escoamento
    ux[mask] = 0
    uy[mask] = 0
    rho[mask] = 1
    return rho, ux, uy


if __name__ == "__main__":
    # teste rapido com um carrinho feito de retangulos
    ny, nx = 80, 240
    mask = np.zeros((ny, nx), dtype=bool)
    mask[38:50, 50:110] = True   # carroceria
    mask[30:38, 65:95] = True    # cabine
    mask[50:54, 55:65] = True    # rodas
    mask[50:54, 95:105] = True

    U = 0.08
    tau = 0.7
    rho, ux, uy = simular(mask, u_entrada=U, tau=tau, iteracoes=20000)

    v = np.sqrt(ux**2 + uy**2)
    print("velocidade maxima:", v.max())
    print("densidade min/max:", rho[~mask].min(), rho[~mask].max())
    print("Reynolds:", U * 60 / tau_para_viscosidade(tau))
