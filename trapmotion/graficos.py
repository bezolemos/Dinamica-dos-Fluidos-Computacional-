"""Figuras que a janela incorpora e que o estudante pode salvar em PNG."""

import numpy as np
from matplotlib.figure import Figure

from aerodinamica import metrics, visualizar
from .integracao import DENSIDADE_AR, U_MALHA

MAPAS = ("Velocidade do ar", "Pressão relativa", "Linhas de corrente", "Vetores do ar", "Silhueta")
COMPARACAO = "Comparar silhuetas"


def graficos_movimento(ensaio):
    fig = Figure(figsize=(9, 5.2), dpi=100, layout="constrained")
    eixos = fig.subplots(2, 2)
    series = (("posicao", "Distância (m)"), ("velocidade", "Velocidade (m/s)"),
              ("aceleracao", "Aceleração (m/s²)"), ("angulo_rad", "Ângulo da mola (°)"))
    # Mesmos históricos da V10 do Projeto 15, agora com os dois cenários sobrepostos.
    for ax, (campo, legenda) in zip(eixos.flat, series):
        for dados, cor, nome, estilo in ((ensaio.ar_parado, "#30718a", "Ar parado", "--"),
                                          (ensaio.com_vento, "#b95332", "Vento informado", "-")):
            hist = dados["historicos"]
            y = np.degrees(hist[campo]) if campo == "angulo_rad" else hist[campo]
            ax.plot(hist["tempo"], y, color=cor, linestyle=estilo, label=nome,
                    marker="o" if len(y) == 1 else None, linewidth=1.8,
                    zorder=3 if estilo == "--" else 2)
        ax.set_ylabel(legenda, fontsize=9)
        ax.set_xlabel("Tempo (s)", fontsize=9)
        ax.grid(alpha=.17)
        ax.tick_params(labelsize=8)
    eixos[0, 0].axhline(ensaio.carro.meta_m, color="#515b60", ls=":", label="Meta")
    eixos[0, 0].legend(fontsize=8, loc="best")
    return fig


def mapa_vento(ensaio, nome, anterior=None):
    t = ensaio.tunel
    if nome == COMPARACAO and anterior is not None:
        return visualizar.comparar_chassis(anterior.tunel.mascara, t.mascara,
                                           nome_1="anterior", nome_2="atual")
    if nome == "Silhueta":
        return visualizar.mostrar_silhueta(t.mascara)
    # A mesma solução 2D adimensional é escalada à velocidade relativa no pico da corrida.
    escala = ensaio.velocidade_relativa_ms / U_MALHA
    ux, uy = t.ux * escala, t.uy * escala
    if nome == "Pressão relativa":
        p = t.metricas["campo_pressao"]
        densidade_entrada = metrics.pressao_entrada(p, t.mascara) / metrics.CS2
        pressao_pa = (p - p[~t.mascara].mean()) * (DENSIDADE_AR / densidade_entrada) * escala**2
        return visualizar.mostrar_pressao(t.rho, t.mascara, pressao=pressao_pa, unidade="Pa")
    if nome == "Velocidade do ar":
        return visualizar.mostrar_velocidade(ux, uy, t.mascara, com_linhas=bool(escala),
                                              com_esteira=False, unidade="m/s")
    if nome in ("Linhas de corrente", "Vetores do ar") and escala == 0:
        fig = visualizar.mostrar_velocidade(ux, uy, t.mascara, com_linhas=False, com_esteira=False, unidade="m/s")
        fig.axes[0].set_title("Ar e carrinho parados · velocidade relativa zero")
        return fig
    if nome == "Linhas de corrente":
        return visualizar.mostrar_linhas_fluxo(ux, uy, t.mascara, densidade=1.3, unidade="m/s")
    if nome == "Vetores do ar":
        return visualizar.mostrar_vetores(ux, uy, t.mascara, unidade="m/s")
    raise ValueError(f"Mapa desconhecido: {nome}")
