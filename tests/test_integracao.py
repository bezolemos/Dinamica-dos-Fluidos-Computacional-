"""Regressões da passagem foto → túnel → força em N → distância da corrida."""
from dataclasses import replace
import math

import numpy as np
from PIL import Image, ImageDraw
import pytest

from aerodinamica import metrics
from trapmotion import movimento
from trapmotion.integracao import DadosCarro, FOTO_EXEMPLO, DENSIDADE_AR, simular_projetos


@pytest.fixture(scope="module")
def ensaio():
    return simular_projetos(FOTO_EXEMPLO, DadosCarro())


def test_foto_da_equipe_move_o_carro_e_alimenta_o_arrasto(ensaio):
    e = ensaio
    d = [r["movimento"]["distancia_total_m"] for r in (e.mecanico, e.ar_parado, e.com_vento)]
    # Referência gerada pelo simulacao.py original do Projeto 15 com as mesmas entradas.
    assert d[0] == pytest.approx(6.2879855282, abs=1e-8)
    assert d[0] > d[1] > d[2] > 0
    assert e.tunel.mascara.any() and e.tunel.convergiu
    assert np.isfinite(e.tunel.ux).all() and np.isfinite(e.tunel.rho).all()
    cd = e.tunel.metricas["coeficiente_arrasto"]
    assert e.com_vento["movimento"]["arrasto_maximo_n"] == pytest.approx(
        .5 * DENSIDADE_AR * cd * e.area_m2 * e.velocidade_relativa_ms**2)
    assert e.perda_m == pytest.approx(d[1] - d[2])
    assert e.perda_pct == pytest.approx(100 * (d[1] - d[2]) / d[1])
    assert e.com_vento["meta"]["resultado"] == "meta_nao_atingida"


def test_ar_parado_e_meta_atingida(ensaio):
    e = simular_projetos(FOTO_EXEMPLO, replace(ensaio.carro, vento_ms=0, meta_m=1),
                        tunel_anterior=ensaio.tunel)
    assert e.tunel is ensaio.tunel
    assert e.perda_m == 0 and e.perda_pct == 0
    assert e.com_vento["meta"]["tempo_meta_atingida_s"] > 0
    assert e.com_vento["movimento"]["arrasto_partida_n"] == 0
    assert e.com_vento["movimento"]["arrasto_maximo_n"] > 0


def test_largura_real_e_usada(ensaio):
    e = simular_projetos(FOTO_EXEMPLO, replace(ensaio.carro, largura_cm=16),
                        tunel_anterior=ensaio.tunel)
    assert e.tunel is ensaio.tunel
    assert e.area_m2 == pytest.approx(2 * ensaio.area_m2)
    assert e.com_vento["movimento"]["distancia_total_m"] < ensaio.com_vento["movimento"]["distancia_total_m"]


def test_vento_pode_impedir_partida(ensaio):
    e = simular_projetos(FOTO_EXEMPLO, replace(ensaio.carro, vento_ms=12), tunel_anterior=ensaio.tunel)
    assert e.com_vento["movimento"]["distancia_total_m"] == 0
    assert e.perda_pct == pytest.approx(100)
    sem_mola = simular_projetos(FOTO_EXEMPLO, replace(ensaio.carro, angulo_partida=0), tunel_anterior=ensaio.tunel)
    assert sem_mola.perda_pct is None


def test_outra_foto_troca_a_geometria(tmp_path, ensaio):
    p = tmp_path / "meu carrinho de teste.png"
    foto = Image.new("RGBA", (260, 120))
    ImageDraw.Draw(foto).polygon([(25, 80), (70, 40), (200, 40), (230, 85)], fill="black")
    foto.save(p)
    e = simular_projetos(p, DadosCarro(), tunel_anterior=ensaio.tunel)
    assert e.tunel is not ensaio.tunel
    assert not np.array_equal(e.tunel.mascara, ensaio.tunel.mascara)
    assert e.tunel.metricas["coeficiente_arrasto"] != pytest.approx(ensaio.tunel.metricas["coeficiente_arrasto"])


def test_inercia_com_arrasto_e_sem_rolamento():
    params = DadosCarro(rolamento=0, corda_cm=2).parametros_movimento()
    com = movimento.executar_simulacao(**params, cd_area_m2=.04, tempo_maximo_simulacao=2)
    sem = movimento.executar_simulacao(**params, tempo_maximo_simulacao=2)
    assert com["movimento"]["tempo_fim_impulsao"] < 2
    assert com["movimento"]["velocidade_final_m_s"] < com["movimento"]["velocidade_fim_impulsao"]
    assert sem["movimento"]["velocidade_final_m_s"] == sem["movimento"]["velocidade_fim_impulsao"]
    assert com["meta"]["resultado"] == "resultado_inconclusivo"


@pytest.mark.parametrize("campo,valor", [("vento_ms", -1), ("comprimento_cm", 0),
                                          ("largura_cm", math.nan), ("massa_g", 0)])
def test_entradas_invalidas(campo, valor):
    with pytest.raises(ValueError):
        simular_projetos(FOTO_EXEMPLO, replace(DadosCarro(), **{campo: valor}))


def test_foto_ausente(tmp_path):
    with pytest.raises(ValueError, match="foto"):
        simular_projetos(tmp_path / "sumiu.png", DadosCarro())


def test_campo_vertical_nao_finito(ensaio):
    t = ensaio.tunel
    ruim = t.uy.copy()
    ruim[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        metrics.calcular_metricas(t.rho, t.ux, ruim, t.mascara, .05, .8)
