# metrics.py - metricas aerodinamicas do Projeto 17
#
# Recebe o resultado do solver (rho, ux, uy) e a mascara do carrinho e
# calcula velocidade, pressao, Reynolds, forca de arrasto e Cd.
# Esse arquivo nao roda simulacao nenhuma.
#
# Tudo em unidades de lattice (dx = dt = 1). Mesmas convencoes do resto do
# projeto: matrizes (ny, nx), linha 0 em cima, vento entrando pela esquerda,
# mask True = carrinho.
#
# Uso:
#   import metrics
#   res = metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=0.05, tau=0.6)
#   metrics.imprimir_metricas(res)
#
#   res["campo_velocidade"] e res["campo_pressao"] podem ir direto pro
#   visualization.py (parametros velocidade= e pressao=).

import numpy as np

CS2 = 1 / 3  # velocidade do som ao quadrado no D2Q9


# ---------------- velocidade ----------------

def calcular_velocidade(ux, uy):
    # v = sqrt(ux^2 + uy^2)
    return np.sqrt(ux**2 + uy**2)


def velocidade_maxima(v, mask):
    mask = np.asarray(mask, dtype=bool)
    return v[~mask].max()   # so conta as celulas de ar


def velocidade_media(v, mask):
    mask = np.asarray(mask, dtype=bool)
    return v[~mask].mean()


# ---------------- pressao ----------------

def calcular_pressao(rho):
    # no D2Q9: p = cs^2 * rho = rho / 3
    return CS2 * np.asarray(rho)


def pressao_entrada(p, mask):
    # pressao do escoamento livre: media na primeira coluna (entrada do vento)
    mask = np.asarray(mask, dtype=bool)
    return p[:, 0][~mask[:, 0]].mean()


# ---------------- viscosidade, Reynolds, Mach ----------------

def viscosidade(tau):
    # nu = (tau - 0.5) / 3
    if tau <= 0.5:
        raise ValueError(f"tau tem que ser maior que 0.5 (recebi {tau})")
    return (tau - 0.5) / 3


def reynolds(U, L, nu):
    # Re = U L / nu
    return U * L / nu


def mach(U):
    # Ma = U / cs. No LBM o ideal e ficar abaixo de 0.3
    return U / np.sqrt(CS2)


# ---------------- tamanho do carrinho ----------------

def comprimento_carro(mask):
    # comprimento na direcao do vento (x), em celulas
    colunas = np.where(np.any(mask, axis=0))[0]
    return colunas.max() - colunas.min() + 1


def area_frontal(mask):
    # Em 2D a "area" frontal e a altura que o vento enxerga: quantas linhas
    # tem pelo menos uma celula de carrinho. Assim o vao entre a carroceria
    # e um aerofolio, por exemplo, nao entra na conta.
    return np.count_nonzero(np.any(mask, axis=1))


# ---------------- forca de arrasto ----------------
#
# Implementei dois jeitos de tirar a forca do resultado do CFD:
#
# 1) volume de controle (o que e usado por padrao): balanco de quantidade
#    de movimento numa caixa em volta do carrinho. A forca e o que o ar
#    "perde" de quantidade de movimento passando pela caixa (a esteira),
#    mais a diferenca de pressao e as tensoes viscosas no contorno.
#    Comparando com o metodo de troca de momento (calculado direto com os
#    f_i dentro do solver) a diferenca deu menos de 0.2%.
#
# 2) superficie: soma a pressao nas faces da frente/tras do carrinho e o
#    atrito nas faces de cima/baixo. Da pra separar arrasto de pressao e de
#    atrito, mas fica uns 5 a 10% abaixo do valor certo, porque a pressao e
#    a velocidade sao lidas a meia celula da parede.

def arrasto_superficie(rho, ux, mask, nu):
    """Retorna (total, parte_pressao, parte_atrito)."""
    mask = np.asarray(mask, dtype=bool)
    ar = ~mask
    p = calcular_pressao(rho)
    dp = p - pressao_entrada(p, mask)

    # frente: ar na esquerda do carrinho empurra ele pra direita
    # tras: ar na direita empurra pra esquerda
    frente = mask[:, 1:] & ar[:, :-1]
    tras = mask[:, :-1] & ar[:, 1:]
    f_pressao = dp[:, :-1][frente].sum() - dp[:, 1:][tras].sum()

    # atrito nas faces de cima e de baixo: tensao = rho * nu * du/dn.
    # No bounce-back a parede fica no meio do caminho, a 0.5 da celula de ar,
    # entao du/dn = ux / 0.5
    cima = mask[1:, :] & ar[:-1, :]
    baixo = mask[:-1, :] & ar[1:, :]
    f_atrito = (rho[:-1][cima] * ux[:-1][cima]).sum() + (rho[1:][baixo] * ux[1:][baixo]).sum()
    f_atrito *= nu / 0.5

    return f_pressao + f_atrito, f_pressao, f_atrito


def caixa_em_volta(mask, margem=None):
    """Caixa (i0, i1, j0, j1) em volta do carrinho, ou None se nao couber na malha."""
    ny, nx = mask.shape
    linhas = np.where(np.any(mask, axis=1))[0]
    colunas = np.where(np.any(mask, axis=0))[0]
    if margem is None:
        margem = max(3, len(linhas) // 2)

    # fica pelo menos 1 celula longe da borda (entrada e saida do solver)
    i0 = max(colunas.min() - margem, 1)
    i1 = min(colunas.max() + margem, nx - 2)
    j0 = max(linhas.min() - margem, 1)
    j1 = min(linhas.max() + margem, ny - 2)

    # e pelo menos 2 celulas longe do carrinho (por causa do np.gradient)
    if i0 > colunas.min() - 2 or i1 < colunas.max() + 2 or j0 > linhas.min() - 2 or j1 < linhas.max() + 2:
        return None
    return i0, i1, j0, j1


def _integral(valores):
    # regra do trapezio com passo 1
    return valores.sum() - 0.5 * (valores[0] + valores[-1])


def _erro_massa(rho, ux, uy, caixa):
    # quanto de massa sai da caixa a mais do que entra (fracao do que entra)
    i0, i1, j0, j1 = caixa
    L = slice(j0, j1 + 1)
    C = slice(i0, i1 + 1)
    entra = _integral(rho[L, i0] * ux[L, i0])
    sai = (_integral(rho[L, i1] * ux[L, i1])
           + _integral(rho[j1, C] * uy[j1, C])
           - _integral(rho[j0, C] * uy[j0, C]))
    return (sai - entra) / entra


def arrasto_volume_controle(rho, ux, uy, mask, nu, margem=None):
    """
    Fx = integral no contorno da caixa de [-p nx - rho ux (u.n) + txx nx + txy ny]
    com txx = 2 rho nu dux/dx e txy = rho nu (dux/dy + duy/dx).

    Retorna (forca, erro_massa). O erro de massa deveria ficar perto de 0
    se o escoamento ja estabilizou. Retorna (None, None) se a caixa nao couber.
    """
    mask = np.asarray(mask, dtype=bool)
    caixa = caixa_em_volta(mask, margem)
    if caixa is None:
        return None, None
    i0, i1, j0, j1 = caixa

    # O sinal de uy depende de como o solver montou o eixo y, e com o sinal
    # errado a forca sai quase o dobro. Mas com o sinal errado a massa nao
    # fecha, entao testo os dois e fico com o que fecha.
    if abs(_erro_massa(rho, ux, -uy, caixa)) < abs(_erro_massa(rho, ux, uy, caixa)):
        uy = -uy

    p = calcular_pressao(rho)
    p = p - pressao_entrada(p, mask)
    txx = 2 * rho * nu * np.gradient(ux, axis=1)
    txy = rho * nu * (np.gradient(ux, axis=0) + np.gradient(uy, axis=1))

    vertical = -p - rho * ux**2 + txx     # o que atravessa as faces esquerda/direita
    horizontal = -rho * ux * uy + txy     # o que atravessa as faces de cima/baixo

    L = slice(j0, j1 + 1)
    C = slice(i0, i1 + 1)
    forca = (_integral(vertical[L, i1]) - _integral(vertical[L, i0])
             + _integral(horizontal[j1, C]) - _integral(horizontal[j0, C]))
    return forca, _erro_massa(rho, ux, uy, caixa)


def forca_arrasto(rho, U, A, cd):
    # Fd = 1/2 rho U^2 A Cd
    return 0.5 * rho * U**2 * A * cd


def coeficiente_arrasto(F, rho, U, A):
    # Cd = Fd / (1/2 rho U^2 A)
    return F / (0.5 * rho * U**2 * A)


# ---------------- juntando tudo ----------------

def calcular_metricas(rho, ux, uy, mask, u_entrada, tau, metodo="volume_controle"):
    """
    Calcula tudo e devolve um dicionario.
    metodo: "volume_controle" (padrao) ou "superficie"
    """
    mask = np.asarray(mask, dtype=bool)
    rho = np.asarray(rho, dtype=float)
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)

    if not (rho.shape == ux.shape == uy.shape == mask.shape):
        raise ValueError("rho, ux, uy e mask tem que ter o mesmo tamanho")
    if not mask.any():
        raise ValueError("a mascara esta vazia, nao tem carrinho")
    if not np.isfinite(rho[~mask]).all() or not np.isfinite(ux[~mask]).all():
        raise ValueError("tem NaN no resultado, a simulacao divergiu")
    if metodo not in ("volume_controle", "superficie"):
        raise ValueError(f"metodo desconhecido: {metodo}")

    nu = viscosidade(tau)
    v = calcular_velocidade(ux, uy)
    p = calcular_pressao(rho)
    rho_ref = pressao_entrada(p, mask) / CS2
    L = comprimento_carro(mask)
    A = area_frontal(mask)
    avisos = []

    f_superficie, f_pressao, f_atrito = arrasto_superficie(rho, ux, mask, nu)
    F, erro_massa = None, None
    if metodo == "volume_controle":
        F, erro_massa = arrasto_volume_controle(rho, ux, uy, mask, nu)
        if F is None:
            avisos.append("nao coube a caixa do volume de controle em volta do carrinho, "
                          "usei o metodo da superficie (fica uns 5-10% abaixo)")
    if F is None:
        F = f_superficie
        metodo = "superficie"
    cd = coeficiente_arrasto(F, rho_ref, u_entrada, A)

    # quanto do arrasto vem da pressao e quanto do atrito (pela superficie)
    if f_superficie > 0:
        pct_pressao = 100 * f_pressao / f_superficie
        pct_atrito = 100 * f_atrito / f_superficie
    else:
        pct_pressao = pct_atrito = None

    if tau < 0.51:
        avisos.append("tau muito perto de 0.5, o resultado pode estar instavel")
    if mach(u_entrada) > 0.3:
        avisos.append("Mach acima de 0.3, diminua a velocidade de entrada")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        avisos.append("o carrinho encosta na borda da malha")
    if erro_massa is not None and abs(erro_massa) > 0.01:
        avisos.append(f"a massa nao fecha na caixa ({100 * erro_massa:+.1f}%), "
                      "o escoamento ainda nao estabilizou")
    if cd < 0:
        avisos.append("Cd negativo, rode mais iteracoes")

    return {
        "u_entrada": u_entrada,
        "velocidade_maxima": velocidade_maxima(v, mask),
        "velocidade_media": velocidade_media(v, mask),
        "pressao_media": p[~mask].mean(),
        "pressao_minima": p[~mask].min(),
        "pressao_maxima": p[~mask].max(),
        "tau": tau,
        "viscosidade": nu,
        "comprimento": L,
        "area_frontal": A,
        "reynolds": reynolds(u_entrada, L, nu),
        "mach": mach(u_entrada),
        "forca_arrasto": F,
        "coeficiente_arrasto": cd,
        "pct_pressao": pct_pressao,
        "pct_atrito": pct_atrito,
        "metodo_arrasto": metodo,
        "erro_massa": erro_massa,
        "avisos": avisos,
        "campo_velocidade": v,
        "campo_pressao": p,
    }


def converter_para_real(res, comprimento_m, velocidade_ms, largura_m=None,
                        rho_ar=1.225, nu_ar=1.5e-5):
    """
    Usa o Cd da simulacao pra estimar a forca no carrinho de verdade.
    Sem a largura a forca sai em N por metro de largura.
    Obs: o Reynolds real e bem maior que o da simulacao, entao o Cd real pode
    ser diferente. E so uma estimativa.
    """
    metros_por_celula = comprimento_m / res["comprimento"]
    altura_m = res["area_frontal"] * metros_por_celula
    area = altura_m * largura_m if largura_m else altura_m
    return {
        "reynolds": velocidade_ms * comprimento_m / nu_ar,
        "area": area,
        "forca_arrasto": forca_arrasto(rho_ar, velocidade_ms, area, res["coeficiente_arrasto"]),
        "unidade": "N" if largura_m else "N/m",
        "velocidade_maxima": res["velocidade_maxima"] * velocidade_ms / res["u_entrada"],
    }


# ---------------- mostrar os resultados ----------------

def imprimir_metricas(res, titulo="Resultados", real=None):
    print("=" * 50)
    print(titulo, "(unidades de lattice)")
    print("=" * 50)
    print(f"Velocidade de entrada   {res['u_entrada']:.4f}")
    print(f"Velocidade maxima       {res['velocidade_maxima']:.4f}")
    print(f"Velocidade media        {res['velocidade_media']:.4f}")
    print(f"Pressao media           {res['pressao_media']:.5f}")
    print(f"Pressao min / max       {res['pressao_minima']:.5f} / {res['pressao_maxima']:.5f}")
    print(f"Viscosidade             {res['viscosidade']:.4f}  (tau = {res['tau']})")
    print(f"Comprimento do carrinho {res['comprimento']} celulas")
    print(f"Area frontal (altura)   {res['area_frontal']} celulas")
    print(f"Reynolds                {res['reynolds']:.1f}")
    print(f"Mach                    {res['mach']:.3f}")
    print(f"Forca de arrasto        {res['forca_arrasto']:.5f}  ({res['metodo_arrasto']})")
    print(f"Cd                      {res['coeficiente_arrasto']:.3f}")
    if res["pct_pressao"] is not None:
        print(f"  pressao / atrito      {res['pct_pressao']:.0f}% / {res['pct_atrito']:.0f}%")
    if real is not None:
        print("-" * 50)
        print("Carrinho real (estimativa)")
        print(f"Reynolds                {real['reynolds']:.0f}")
        print(f"Velocidade maxima       {real['velocidade_maxima']:.2f} m/s")
        print(f"Forca de arrasto        {real['forca_arrasto']:.4f} {real['unidade']}")
    for aviso in res["avisos"]:
        print("AVISO:", aviso)
    print("=" * 50)


def comparar_resultados(r1, r2, nome1="Chassi 1", nome2="Chassi 2"):
    linhas = [
        ("velocidade_maxima", "Velocidade maxima"),
        ("velocidade_media", "Velocidade media"),
        ("reynolds", "Reynolds"),
        ("forca_arrasto", "Forca de arrasto"),
        ("coeficiente_arrasto", "Cd"),
        ("pct_pressao", "% arrasto de pressao"),
    ]
    print(f"{'':24}{nome1:>12}{nome2:>12}{'diferenca':>11}")
    for chave, nome in linhas:
        a, b = r1[chave], r2[chave]
        if a is None or b is None:
            continue
        dif = f"{100 * (b - a) / abs(a):+.1f}%" if a != 0 else "-"
        print(f"{nome:24}{a:12.4g}{b:12.4g}{dif:>11}")

    cd1, cd2 = r1["coeficiente_arrasto"], r2["coeficiente_arrasto"]
    melhor = nome1 if cd1 < cd2 else nome2
    print(f"\nMenor arrasto: {melhor} ({100 * abs(cd1 - cd2) / max(cd1, cd2):.1f}% menos Cd)")
    if abs(r1["reynolds"] - r2["reynolds"]) > 0.05 * r1["reynolds"]:
        print("obs: os Reynolds sao diferentes, entao a comparacao do Cd nao e totalmente justa")


if __name__ == "__main__":
    # teste rapido: roda o solver em dois carrinhos de retangulos e compara.
    # O Cd sai alto (~6) porque a malha e pequena: o carrinho ocupa quase 30%
    # da altura do tunel e o Reynolds e baixo. Com a malha de verdade cai bastante.
    import LBM

    ny, nx = 60, 180
    U, tau = 0.08, 0.8

    carro1 = np.zeros((ny, nx), dtype=bool)
    carro1[28:36, 40:80] = True      # carroceria
    carro1[22:28, 52:70] = True      # cabine
    carro1[36:39, 44:50] = True      # rodas
    carro1[36:39, 70:76] = True

    carro2 = carro1.copy()
    carro2[22:28, 70:80] = True      # cabine indo ate a traseira (tipo van)

    resultados = []
    for nome, mask in (("carro 1", carro1), ("carro 2", carro2)):
        print("simulando", nome)
        rho, ux, uy = LBM.simular(mask, u_entrada=U, tau=tau, iteracoes=10000, mostrar_progresso=False)
        resultados.append(calcular_metricas(rho, ux, uy, mask, u_entrada=U, tau=tau))

    real = converter_para_real(resultados[0], comprimento_m=0.30, velocidade_ms=2.0, largura_m=0.10)
    imprimir_metricas(resultados[0], "carro 1", real=real)
    print()
    comparar_resultados(resultados[0], resultados[1], "carro 1", "carro 2")
