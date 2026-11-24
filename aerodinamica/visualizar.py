"""
visualization.py — Visualização do túnel de vento (Projeto 17)

Este módulo NÃO calcula CFD. Ele recebe os resultados do solver
(rho, ux, uy) e a máscara do carrinho (mask) e apenas os desenha.

Convenções (iguais às do restante do projeto):
    - Todas as matrizes têm formato (ny, nx): linha = y, coluna = x.
    - A linha 0 fica no TOPO, como em uma imagem (mesma convenção do geometry.py).
    - O vento entra pela ESQUERDA e sai pela DIREITA (esteira fica à direita do carrinho).
    - mask é booleana: True = carrinho, False = ar.

Uso a partir do main.py:
    from aerodinamica import visualizar as viz
    viz.gerar_todos_os_graficos(rho, ux, uy, mask, pasta_saida="resultados")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _preparar_mascara(mask):
    """Garante que a máscara é uma matriz booleana."""
    return np.asarray(mask).astype(bool)


def calcular_velocidade(ux, uy):
    """Magnitude da velocidade: v = sqrt(ux² + uy²)."""
    return np.sqrt(np.asarray(ux) ** 2 + np.asarray(uy) ** 2)


def calcular_pressao(rho):
    """
    Pressão no D2Q9: p = cs² * rho, com cs² = 1/3.

    Se o metrics.py já entregar a pressão pronta, passe-a diretamente
    para mostrar_pressao(..., pressao=...) e esta função não é usada.
    """
    return np.asarray(rho) / 3.0


def _campo_com_mascara(campo, mask):
    """Esconde (mascara) as células do carrinho para elas não 'sujarem' a escala de cores."""
    return np.ma.masked_where(mask, campo)


def _cmap_com_carro(nome, cor_carro="0.35"):
    """Cópia do colormap em que as células mascaradas aparecem em cinza."""
    cmap = plt.get_cmap(nome).copy()
    cmap.set_bad(cor_carro)
    return cmap


def _desenhar_contorno_carro(ax, mask, cor="black", espessura=1.5):
    """Desenha o contorno do carrinho. Só desenha se houver carrinho na máscara."""
    if mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors=cor, linewidths=espessura)


def _limites_do_carro(mask):
    """Retorna (x_min, x_max, y_min, y_max) do carrinho, ou None se a máscara estiver vazia."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return xs.min(), xs.max(), ys.min(), ys.max()


def _grade(mask):
    """Grade X, Y no mesmo sistema de coordenadas do imshow."""
    ny, nx = mask.shape
    return np.meshgrid(np.arange(nx), np.arange(ny))


def _finalizar(fig, titulo, salvar_em=None, dpi=200):
    """Título, layout e salvamento. Não chama plt.show() (isso fica para o final / main.py)."""
    ax = fig.axes[0]
    ax.set_title(titulo)
    ax.set_xlabel("x (células)")
    ax.set_ylabel("y (células)")
    fig.tight_layout()
    if salvar_em is not None:
        salvar_em = Path(salvar_em)
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=dpi, bbox_inches="tight")
    return fig


# ============================================================
# ESTEIRA (WAKE)
# ============================================================

def desenhar_esteira(ax, mask, comprimento_rel=1.5, margem_rel=0.15, cor="red"):
    """
    Destaca a região de esteira ATRÁS do carrinho (lado direito).

    A região é calculada a partir da própria máscara (e não mais fixa em x=4..5):
        - começa na traseira do carrinho;
        - tem comprimento = comprimento_rel * comprimento do carrinho;
        - tem altura do carrinho + margem_rel de folga em cima e embaixo.
    """
    ny, nx = mask.shape
    limites = _limites_do_carro(mask)
    if limites is None:
        return

    x_min, x_max, y_min, y_max = limites
    comp_carro = x_max - x_min + 1
    alt_carro = y_max - y_min + 1

    x_ini = x_max + 0.5
    x_fim = min(x_ini + comprimento_rel * comp_carro, nx - 0.5)
    folga = margem_rel * alt_carro
    y_ini = max(y_min - folga - 0.5, -0.5)
    y_fim = min(y_max + folga + 0.5, ny - 0.5)

    ax.add_patch(
        Rectangle(
            (x_ini, y_ini),
            x_fim - x_ini,
            y_fim - y_ini,
            facecolor=cor,
            edgecolor=cor,
            alpha=0.15,
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.text(
        x_ini + 0.02 * (x_fim - x_ini),
        y_ini + 0.08 * (y_fim - y_ini),
        "Esteira",
        color=cor,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


# ============================================================
# 1) SILHUETA DO CARRINHO NA MALHA
# ============================================================

def mostrar_silhueta(mask, salvar_em=None):
    """Mostra a máscara (carrinho x ar) como o solver a enxerga."""
    mask = _preparar_mascara(mask)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(mask, cmap="gray_r", interpolation="nearest")
    _desenhar_contorno_carro(ax, mask, cor="red")

    return _finalizar(fig, "Silhueta do carrinho na malha", salvar_em)


# ============================================================
# 2) CAMPO DE VELOCIDADE
# ============================================================

def mostrar_velocidade(ux, uy, mask, velocidade=None, com_linhas=True,
                       com_esteira=True, salvar_em=None, unidade="unid. de lattice"):
    """
    Mapa de cores da velocidade resultante, com linhas de corrente por cima.

    velocidade: opcional. Se o metrics.py já calculou, passe aqui;
                caso contrário é calculada como sqrt(ux² + uy²).
    """
    mask = _preparar_mascara(mask)
    if velocidade is None:
        velocidade = calcular_velocidade(ux, uy)

    fig, ax = plt.subplots(figsize=(11, 5))

    imagem = ax.imshow(
        _campo_com_mascara(velocidade, mask),
        cmap=_cmap_com_carro("viridis"),
        interpolation="nearest",
    )
    fig.colorbar(imagem, ax=ax, label=f"Velocidade ({unidade})", shrink=0.85)

    if com_linhas:
        X, Y = _grade(mask)
        ax.streamplot(
            X, Y,
            _campo_com_mascara(ux, mask),
            _campo_com_mascara(uy, mask),
            color="white", linewidth=0.6, density=1.4, arrowsize=0.8,
        )

    _desenhar_contorno_carro(ax, mask)
    if com_esteira:
        desenhar_esteira(ax, mask)

    return _finalizar(fig, "Campo de velocidade", salvar_em)


# ============================================================
# 3) CAMPO DE PRESSÃO
# ============================================================

def mostrar_pressao(rho, mask, pressao=None, relativa=True, salvar_em=None, unidade="unid. de lattice"):
    """
    Mapa de pressão.

    rho:      densidade vinda do solver (usada se 'pressao' não for passada,
              com p = rho/3 conforme o D2Q9).
    pressao:  opcional, caso o metrics.py já tenha calculado.
    relativa: se True, mostra p - p_médio_do_ar com colormap divergente.
              No LBM a densidade fica perto de 1, então a variação absoluta
              é minúscula e o gráfico ficaria "chapado"; a versão relativa
              deixa claro onde há sobrepressão (frente) e sucção (topo/traseira).
    """
    mask = _preparar_mascara(mask)
    if pressao is None:
        pressao = calcular_pressao(rho)
    pressao = np.asarray(pressao, dtype=float)

    if relativa:
        campo = pressao - pressao[~mask].mean()
        limite = np.abs(campo[~mask]).max()
        limite = limite if limite > 0 else 1.0
        cmap = _cmap_com_carro("coolwarm")
        kwargs = dict(vmin=-limite, vmax=limite)
        rotulo = f"Pressão − pressão média ({unidade})"
    else:
        campo = pressao
        cmap = _cmap_com_carro("plasma")
        kwargs = {}
        rotulo = f"Pressão ({unidade})"

    fig, ax = plt.subplots(figsize=(11, 5))
    imagem = ax.imshow(
        _campo_com_mascara(campo, mask), cmap=cmap, interpolation="nearest", **kwargs
    )
    fig.colorbar(imagem, ax=ax, label=rotulo, shrink=0.85)
    _desenhar_contorno_carro(ax, mask)

    return _finalizar(fig, "Campo de pressão", salvar_em)


# ============================================================
# 4) LINHAS DE FLUXO (LINHAS DE CORRENTE)
# ============================================================

def mostrar_linhas_fluxo(ux, uy, mask, velocidade=None, densidade=2.0, salvar_em=None, unidade="unid. de lattice"):
    """Linhas de corrente coloridas pela velocidade local."""
    mask = _preparar_mascara(mask)
    if velocidade is None:
        velocidade = calcular_velocidade(ux, uy)

    X, Y = _grade(mask)
    fig, ax = plt.subplots(figsize=(11, 5))

    # Fundo cinza claro só para o carrinho aparecer preenchido
    ax.imshow(mask, cmap="Greys", alpha=0.6, interpolation="nearest")

    linhas = ax.streamplot(
        X, Y,
        _campo_com_mascara(ux, mask),
        _campo_com_mascara(uy, mask),
        color=_campo_com_mascara(velocidade, mask),
        cmap="viridis",
        density=densidade,
        linewidth=1.0,
        arrowsize=0.9,
    )
    fig.colorbar(linhas.lines, ax=ax, label=f"Velocidade ({unidade})", shrink=0.85)
    _desenhar_contorno_carro(ax, mask, cor="red")

    ax.set_xlim(-0.5, mask.shape[1] - 0.5)
    ax.set_ylim(mask.shape[0] - 0.5, -0.5)  # mantém linha 0 no topo, como o imshow

    return _finalizar(fig, "Linhas de fluxo", salvar_em)


# ============================================================
# 5) VETORES DE VELOCIDADE
# ============================================================

def mostrar_vetores(ux, uy, mask, velocidade=None, passo=None, salvar_em=None, unidade="unid. de lattice"):
    """
    Campo vetorial (quiver) com setas coloridas pela velocidade.

    passo: mostra 1 seta a cada 'passo' células. Se None, é escolhido
           automaticamente para ficar com ~40 setas na maior dimensão.
    """
    mask = _preparar_mascara(mask)
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    if velocidade is None:
        velocidade = calcular_velocidade(ux, uy)

    ny, nx = mask.shape
    if passo is None:
        passo = max(1, max(nx, ny) // 40)

    X, Y = _grade(mask)
    sel = (slice(None, None, passo), slice(None, None, passo))
    fora_do_carro = ~mask[sel]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(mask, cmap="Greys", alpha=0.6, interpolation="nearest")

    setas = ax.quiver(
        X[sel][fora_do_carro],
        Y[sel][fora_do_carro],
        ux[sel][fora_do_carro],
        uy[sel][fora_do_carro],
        velocidade[sel][fora_do_carro],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    fig.colorbar(setas, ax=ax, label=f"Velocidade ({unidade})", shrink=0.85)
    _desenhar_contorno_carro(ax, mask, cor="red")

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)

    return _finalizar(fig, "Vetores de velocidade", salvar_em)


# ============================================================
# 6) PAINEL COMPLETO (4 gráficos em uma figura)
# ============================================================

def mostrar_painel(rho, ux, uy, mask, titulo="Resultado da simulação", salvar_em=None):
    """Resumo visual em uma única figura: velocidade, pressão, linhas de fluxo e vetores."""
    mask = _preparar_mascara(mask)
    velocidade = calcular_velocidade(ux, uy)
    pressao = calcular_pressao(rho)
    ny, nx = mask.shape
    X, Y = _grade(mask)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # Velocidade
    ax = axes[0, 0]
    im = ax.imshow(_campo_com_mascara(velocidade, mask),
                   cmap=_cmap_com_carro("viridis"), interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Velocidade")
    _desenhar_contorno_carro(ax, mask)
    desenhar_esteira(ax, mask)
    ax.set_title("Velocidade")

    # Pressão relativa
    ax = axes[0, 1]
    rel = pressao - pressao[~mask].mean()
    lim = max(np.abs(rel[~mask]).max(), 1e-12)
    im = ax.imshow(_campo_com_mascara(rel, mask), cmap=_cmap_com_carro("coolwarm"),
                   vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ pressão")
    _desenhar_contorno_carro(ax, mask)
    ax.set_title("Pressão (relativa)")

    # Linhas de fluxo
    ax = axes[1, 0]
    ax.imshow(mask, cmap="Greys", alpha=0.6, interpolation="nearest")
    ax.streamplot(X, Y, _campo_com_mascara(ux, mask), _campo_com_mascara(uy, mask),
                  color="tab:blue", density=1.6, linewidth=0.8, arrowsize=0.8)
    _desenhar_contorno_carro(ax, mask, cor="red")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_title("Linhas de fluxo")

    # Vetores
    ax = axes[1, 1]
    passo = max(1, max(nx, ny) // 30)
    sel = (slice(None, None, passo), slice(None, None, passo))
    ok = ~mask[sel]
    ax.imshow(mask, cmap="Greys", alpha=0.6, interpolation="nearest")
    ax.quiver(X[sel][ok], Y[sel][ok], np.asarray(ux)[sel][ok], np.asarray(uy)[sel][ok],
              angles="xy", scale_units="xy", scale=None, color="tab:blue")
    _desenhar_contorno_carro(ax, mask, cor="red")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_title("Vetores")

    for ax in axes.flat:
        ax.set_xlabel("x (células)")
        ax.set_ylabel("y (células)")

    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    if salvar_em is not None:
        salvar_em = Path(salvar_em)
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=200, bbox_inches="tight")
    return fig


# ============================================================
# 7) COMPARAÇÃO ENTRE DOIS CHASSIS
# ============================================================

def comparar_chassis(mask_1, mask_2, resultado_1=None, resultado_2=None,
                     nome_1="Chassi 1", nome_2="Chassi 2", salvar_em=None):
    """
    Compara dois chassis lado a lado.

    Só com as máscaras: mostra as duas silhuetas e onde elas diferem.

    Com resultado_1 / resultado_2 (dicionários {"ux": ..., "uy": ...}, opcionalmente
    "rho"): mostra também o campo de velocidade de cada um NA MESMA ESCALA DE CORES
    (sem isso a comparação visual engana) e a diferença de velocidade entre os dois.
    As duas simulações precisam ter a mesma malha.
    """
    mask_1 = _preparar_mascara(mask_1)
    mask_2 = _preparar_mascara(mask_2)
    if mask_1.shape != mask_2.shape:
        raise ValueError(
            f"As máscaras precisam ter o mesmo tamanho: {mask_1.shape} x {mask_2.shape}"
        )

    com_campos = resultado_1 is not None and resultado_2 is not None

    if not com_campos:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].imshow(mask_1, cmap="gray_r", interpolation="nearest")
        axes[0].set_title(nome_1)
        axes[1].imshow(mask_2, cmap="gray_r", interpolation="nearest")
        axes[1].set_title(nome_2)

        # Diferença: verde = só no chassi 1, vermelho = só no chassi 2, cinza = nos dois
        dif = np.zeros(mask_1.shape + (3,))
        dif[mask_1 & mask_2] = (0.5, 0.5, 0.5)
        dif[mask_1 & ~mask_2] = (0.1, 0.7, 0.2)
        dif[~mask_1 & mask_2] = (0.85, 0.15, 0.15)
        dif[~mask_1 & ~mask_2] = (1, 1, 1)
        axes[2].imshow(dif, interpolation="nearest")
        axes[2].set_title(f"Diferença (verde: só {nome_1} | vermelho: só {nome_2})", fontsize=9)

        for ax in axes:
            ax.set_xlabel("x (células)")
            ax.set_ylabel("y (células)")
    else:
        v1 = calcular_velocidade(resultado_1["ux"], resultado_1["uy"])
        v2 = calcular_velocidade(resultado_2["ux"], resultado_2["uy"])
        if v1.shape != v2.shape:
            raise ValueError("As duas simulações precisam ter a mesma malha.")

        vmax = max(v1[~mask_1].max(), v2[~mask_2].max())
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

        for ax, v, m, nome in ((axes[0], v1, mask_1, nome_1), (axes[1], v2, mask_2, nome_2)):
            im = ax.imshow(_campo_com_mascara(v, m), cmap=_cmap_com_carro("viridis"),
                           vmin=0, vmax=vmax, interpolation="nearest")
            _desenhar_contorno_carro(ax, m)
            desenhar_esteira(ax, m)
            ax.set_title(f"Velocidade — {nome}")
        fig.colorbar(im, ax=axes[:2], shrink=0.8, label="Velocidade (mesma escala)")

        # Diferença só onde nenhum dos dois é carrinho
        dif = np.ma.masked_where(mask_1 | mask_2, v2 - v1)
        lim = max(np.abs(dif).max(), 1e-12)
        im = axes[2].imshow(dif, cmap=_cmap_com_carro("RdBu_r"), vmin=-lim, vmax=lim,
                            interpolation="nearest")
        fig.colorbar(im, ax=axes[2], shrink=0.8, label=f"{nome_2} − {nome_1}")
        axes[2].set_title("Diferença de velocidade")

        for ax in axes:
            ax.set_xlabel("x (células)")
            ax.set_ylabel("y (células)")

    if salvar_em is not None:
        salvar_em = Path(salvar_em)
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=200, bbox_inches="tight")
    return fig


# ============================================================
# GERA TUDO DE UMA VEZ (é isto que o main.py deve chamar)
# ============================================================

def gerar_todos_os_graficos(rho, ux, uy, mask, pasta_saida="resultados", mostrar=False):
    """Gera e salva todos os gráficos em 'pasta_saida'. Retorna a lista de arquivos."""
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    mask = _preparar_mascara(mask)

    arquivos = {
        "silhueta.png": lambda p: mostrar_silhueta(mask, salvar_em=p),
        "campo_velocidade.png": lambda p: mostrar_velocidade(ux, uy, mask, salvar_em=p),
        "campo_pressao.png": lambda p: mostrar_pressao(rho, mask, salvar_em=p),
        "linhas_fluxo.png": lambda p: mostrar_linhas_fluxo(ux, uy, mask, salvar_em=p),
        "vetores_velocidade.png": lambda p: mostrar_vetores(ux, uy, mask, salvar_em=p),
        "painel_completo.png": lambda p: mostrar_painel(rho, ux, uy, mask, salvar_em=p),
    }

    salvos = []
    for nome, criar in arquivos.items():
        caminho = pasta / nome
        fig = criar(caminho)
        salvos.append(caminho)
        if not mostrar:
            plt.close(fig)  # libera memória quando não for exibir

    if mostrar:
        plt.show()
    return salvos


# ============================================================
# DADOS FICTÍCIOS PARA TESTE (só rodam com: python visualization.py)
# ============================================================

def _gerar_dados_teste(nx=200, ny=80, altura_carro=18, comprimento_carro=50, x0=50):
    """
    Campo sintético parecido com um escoamento ao redor de um obstáculo:
    escoamento potencial em volta de um cilindro + déficit de velocidade
    na esteira. NÃO é física de verdade — serve apenas para testar os gráficos.
    """
    U = 0.1
    y_idx, x_idx = np.mgrid[0:ny, 0:nx].astype(float)

    mask = np.zeros((ny, nx), dtype=bool)
    yc = ny // 2
    mask[yc - altura_carro // 2: yc + altura_carro // 2, x0: x0 + comprimento_carro] = True

    xc = x0 + comprimento_carro / 2
    a = altura_carro / 2 + 2
    dx, dy = x_idx - xc, y_idx - yc
    r2 = np.maximum(dx ** 2 + dy ** 2, a ** 2)

    ux = U * (1 - a ** 2 * (dx ** 2 - dy ** 2) / r2 ** 2)
    uy = -U * a ** 2 * 2 * dx * dy / r2 ** 2

    # Déficit de velocidade na esteira (atrás do carrinho)
    atras = np.clip((x_idx - (x0 + comprimento_carro)) / 40.0, 0, None)
    deficit = 0.6 * np.exp(-atras) * np.exp(-(dy / (altura_carro * 0.8)) ** 2) * (x_idx > x0 + comprimento_carro)
    ux = ux * (1 - deficit)

    ux[mask] = 0.0
    uy[mask] = 0.0

    v2 = ux ** 2 + uy ** 2
    rho = 1.0 + 3.0 * (U ** 2 - v2)  # Bernoulli "de brincadeira": mais velocidade, menos pressão
    return rho, ux, uy, mask


if __name__ == "__main__":
    rho, ux, uy, mask = _gerar_dados_teste()

    # Segundo chassi: mais alto e mais curto, só para testar a comparação
    rho2, ux2, uy2, mask2 = _gerar_dados_teste(altura_carro=26, comprimento_carro=40, x0=55)

    gerar_todos_os_graficos(rho, ux, uy, mask, pasta_saida="resultados_teste")

    comparar_chassis(mask, mask2, salvar_em="resultados_teste/comparacao_mascaras.png")
    comparar_chassis(
        mask, mask2,
        resultado_1={"ux": ux, "uy": uy},
        resultado_2={"ux": ux2, "uy": uy2},
        salvar_em="resultados_teste/comparacao_campos.png",
    )

    print("Gráficos de teste salvos em ./resultados_teste/")
    plt.show()
