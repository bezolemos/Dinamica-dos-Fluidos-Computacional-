# Testes do metrics.py. Rodar com:  pytest tests/ -v

import numpy as np
import pytest

from aerodinamica import lbm as LBM
from aerodinamica import metrics

U = 0.05


def bloco(ny=40, nx=100, y0=15, y1=24, x0=30, x1=59):
    # carrinho retangular nas linhas y0..y1 e colunas x0..x1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def vento_uniforme(mask, u=U):
    rho = np.ones(mask.shape)
    ux = np.where(mask, 0.0, u)
    uy = np.zeros(mask.shape)
    return rho, ux, uy


# --- contas simples ---

def test_velocidade():
    v = metrics.calcular_velocidade(np.array([3.0, 0.0]), np.array([4.0, -2.0]))
    assert v == pytest.approx([5.0, 2.0])


def test_max_e_media_nao_contam_o_carrinho():
    v = np.array([[1.0, 3.0, 100.0]])
    mask = np.array([[False, False, True]])
    assert metrics.velocidade_maxima(v, mask) == 3.0
    assert metrics.velocidade_media(v, mask) == 2.0


def test_pressao():
    rho = np.array([1.0, 0.97, 1.03])
    assert metrics.calcular_pressao(rho) == pytest.approx(rho / 3)


def test_viscosidade_reynolds_mach():
    assert metrics.viscosidade(0.8) == pytest.approx(0.1)
    assert metrics.reynolds(0.05, 60, 0.01) == pytest.approx(300)
    assert metrics.reynolds(2.0, 0.30, 1.5e-5) == pytest.approx(40000)  # carrinho real
    assert metrics.mach(1 / np.sqrt(3)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.viscosidade(0.5)


def test_comprimento_e_area_frontal():
    mask = bloco(y0=15, y1=20, x0=30, x1=59)
    assert metrics.comprimento_carro(mask) == 30
    assert metrics.area_frontal(mask) == 6
    mask[21:24, 32:36] = True   # roda embaixo: +3 linhas
    mask[10:12, 40:45] = True   # aerofolio solto em cima: +2 linhas, o vao nao conta
    assert metrics.area_frontal(mask) == 11


def test_cd_e_forca():
    F = metrics.forca_arrasto(1.225, 2.0, 0.01, 0.8)
    assert F == pytest.approx(0.5 * 1.225 * 4 * 0.01 * 0.8)
    assert metrics.coeficiente_arrasto(F, 1.225, 2.0, 0.01) == pytest.approx(0.8)


# --- arrasto com campos montados na mao ---

def test_arrasto_de_pressao():
    # pressao maior na frente e menor atras de um bloco com 10 linhas
    mask = bloco()
    rho = np.ones(mask.shape)
    rho[:, :30] += 0.003
    rho[:, 60:] -= 0.003
    zero = np.zeros(mask.shape)
    total, f_p, f_a = metrics.arrasto_superficie(rho, zero, mask, nu=0.01)
    # a pressao de referencia e a da entrada (1.003/3); a diferenca frente-tras e 0.006/3
    assert f_p == pytest.approx(10 * 0.006 / 3)
    assert f_a == 0


def test_pressao_igual_em_tudo_nao_da_forca():
    mask = bloco()
    rho, ux, uy = vento_uniforme(mask)
    rho *= 1.02
    _, f_p, _ = metrics.arrasto_superficie(rho, ux, mask, nu=0.01)
    assert f_p == pytest.approx(0, abs=1e-12)


def test_atrito():
    # vento U encostado nas faces de cima e de baixo (30 celulas cada)
    mask = bloco()
    rho, ux, uy = vento_uniforme(mask)
    _, _, f_a = metrics.arrasto_superficie(rho, ux, mask, nu=0.02)
    assert f_a == pytest.approx(2 * 30 * 0.02 * U / 0.5)


def test_volume_controle_sem_perturbacao_da_zero():
    mask = bloco()
    rho, ux, uy = vento_uniforme(mask)
    ux[:] = U
    F, erro = metrics.arrasto_volume_controle(rho, ux, uy, mask, nu=0.01)
    assert F == pytest.approx(0, abs=1e-12)
    assert erro == pytest.approx(0, abs=1e-12)


def test_volume_controle_esteira():
    # o ar sai mais devagar atras do carrinho: F = soma de (U^2 - u^2) na saida
    mask = bloco()
    rho, ux, uy = vento_uniforme(mask)
    ux[:] = U
    i0, i1, j0, j1 = metrics.caixa_em_volta(mask, margem=4)
    ux[15:25, i1 - 1:] = 0.6 * U
    F, _ = metrics.arrasto_volume_controle(rho, ux, uy, mask, nu=0, margem=4)
    assert F == pytest.approx(10 * (U**2 - (0.6 * U)**2))


def test_caixa_nao_cabe():
    assert metrics.caixa_em_volta(bloco(y0=1, y1=10)) is None


# --- calcular_metricas ---

def test_avisos():
    mask = bloco(y0=0, y1=9)   # encostado no topo
    rho, ux, uy = vento_uniforme(mask, u=0.25)
    res = metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=0.25, tau=0.505)
    texto = " ".join(res["avisos"])
    assert "Mach" in texto and "tau" in texto and "borda" in texto
    assert res["metodo_arrasto"] == "superficie"   # a caixa nao coube


def test_erros():
    mask = bloco()
    rho, ux, uy = vento_uniforme(mask)
    ux[5, 5] = np.nan
    with pytest.raises(ValueError, match="divergiu"):
        metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=U, tau=0.6)
    with pytest.raises(ValueError):
        metrics.calcular_metricas(rho, ux[:, 1:], uy, mask, u_entrada=U, tau=0.6)
    with pytest.raises(ValueError):
        metrics.calcular_metricas(rho, ux, uy, np.zeros_like(mask), u_entrada=U, tau=0.6)


def test_converter_para_real():
    # carrinho de 50 x 18 celulas que na vida real tem 50 cm
    res = {"comprimento": 50, "area_frontal": 18, "coeficiente_arrasto": 1.2,
           "velocidade_maxima": 0.06, "u_entrada": U}
    real = metrics.converter_para_real(res, comprimento_m=0.5, velocidade_ms=2.0, largura_m=0.1)
    assert real["area"] == pytest.approx(0.18 * 0.1)
    assert real["forca_arrasto"] == pytest.approx(0.5 * 1.225 * 4 * 0.018 * 1.2)
    assert real["velocidade_maxima"] == pytest.approx(2.4)
    assert real["unidade"] == "N"
    assert metrics.converter_para_real(res, 0.5, 2.0)["unidade"] == "N/m"


# --- com o solver de verdade ---

@pytest.fixture(scope="module")
def simulacao():
    mask = np.zeros((32, 96), dtype=bool)
    mask[12:20, 24:40] = True   # carroceria
    mask[9:12, 28:35] = True    # cabine
    tau, u = 0.8, 0.1
    rho, ux, uy = LBM.simular(mask, u_entrada=u, tau=tau, iteracoes=4000, mostrar_progresso=False)
    return rho, ux, uy, mask, u, tau


def test_volume_controle_nao_depende_do_tamanho_da_caixa(simulacao):
    rho, ux, uy, mask, u, tau = simulacao
    nu = metrics.viscosidade(tau)
    forcas = [metrics.arrasto_volume_controle(rho, ux, uy, mask, nu, margem=m)[0] for m in (3, 5, 8)]
    assert max(forcas) - min(forcas) < 0.01 * forcas[0]


def test_sinal_de_uy_e_descoberto_sozinho(simulacao):
    rho, ux, uy, mask, u, tau = simulacao
    nu = metrics.viscosidade(tau)
    F1, _ = metrics.arrasto_volume_controle(rho, ux, uy, mask, nu)
    F2, _ = metrics.arrasto_volume_controle(rho, ux, -uy, mask, nu)
    assert F2 == pytest.approx(F1)


def test_metricas_da_simulacao(simulacao):
    rho, ux, uy, mask, u, tau = simulacao
    res = metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=u, tau=tau)
    assert res["metodo_arrasto"] == "volume_controle"
    assert abs(res["erro_massa"]) < 0.01
    assert res["avisos"] == []
    assert res["coeficiente_arrasto"] > 0
    assert res["velocidade_maxima"] > u   # o ar acelera passando por cima do carrinho
    assert res["pct_pressao"] + res["pct_atrito"] == pytest.approx(100)

    # o metodo da superficie fica um pouco abaixo
    sup = metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=u, tau=tau, metodo="superficie")
    assert 0.85 * res["forca_arrasto"] < sup["forca_arrasto"] < res["forca_arrasto"]


def test_massa_nao_fecha_com_campo_errado(simulacao):
    rho, ux, uy, mask, u, tau = simulacao
    ux = ux.copy()
    ux[:, 42:] *= 0.8   # some 20% do ar atras do carrinho
    res = metrics.calcular_metricas(rho, ux, uy, mask, u_entrada=u, tau=tau)
    assert any("massa" in a for a in res["avisos"])
