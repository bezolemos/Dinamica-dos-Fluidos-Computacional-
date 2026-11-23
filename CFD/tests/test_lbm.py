# Testes do LBM.py. Rodar com:  pytest tests/ -v

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LBM  # noqa: E402


def test_pesos_e_direcoes():
    assert LBM.w.sum() == pytest.approx(1)
    # a direcao oposta tem que ser mesmo a contraria
    assert (LBM.ex[LBM.oposto] == -LBM.ex).all()
    assert (LBM.ey[LBM.oposto] == -LBM.ey).all()


def test_equilibrio_devolve_rho_e_u():
    rho = np.full((3, 4), 1.02)
    ux = np.full((3, 4), 0.05)
    uy = np.full((3, 4), -0.02)
    r, u, v = LBM.macroscopicas(LBM.equilibrio(rho, ux, uy))
    assert r == pytest.approx(rho)
    assert u == pytest.approx(ux)
    assert v == pytest.approx(uy)


def test_viscosidade_tau():
    assert LBM.tau_para_viscosidade(0.8) == pytest.approx(0.1)
    assert LBM.viscosidade_para_tau(0.1) == pytest.approx(0.8)


def test_streaming_anda_uma_celula():
    f = np.zeros((9, 5, 5))
    f[:, 2, 2] = 1
    f = LBM.streaming(f)
    for i in range(9):
        assert f[i, 2 + LBM.ey[i], 2 + LBM.ex[i]] == 1


def test_sem_carrinho_o_vento_continua_igual():
    mask = np.zeros((20, 60), dtype=bool)
    rho, ux, uy = LBM.simular(mask, u_entrada=0.05, tau=0.6, iteracoes=300, mostrar_progresso=False)
    assert ux == pytest.approx(0.05, abs=1e-6)
    assert uy == pytest.approx(0, abs=1e-6)
    assert rho == pytest.approx(1, abs=1e-6)


def test_carrinho_freia_o_ar():
    mask = np.zeros((40, 120), dtype=bool)
    mask[16:24, 30:50] = True
    rho, ux, uy = LBM.simular(mask, u_entrada=0.05, tau=0.8, iteracoes=2000, mostrar_progresso=False)
    assert (ux[mask] == 0).all() and (uy[mask] == 0).all()
    assert ux[20, 51] < 0.2 * 0.05          # logo atras do carrinho o ar quase para
    assert ux[3, 40] > 0.05                 # por cima ele acelera
    assert rho[20, 29] > rho[20, 51]        # mais pressao na frente do que atras
    # a massa que entra e a que sai tem que bater
    entra = (rho[:, 1] * ux[:, 1]).sum()
    sai = (rho[:, -2] * ux[:, -2]).sum()
    assert sai == pytest.approx(entra, rel=1e-3)


def test_tau_invalido():
    with pytest.raises(ValueError):
        LBM.simular(np.zeros((10, 10), dtype=bool), tau=0.5)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # o numpy reclama dos NaN
def test_divergencia_vira_erro():
    mask = np.zeros((20, 60), dtype=bool)
    mask[8:12, 15:25] = True
    with pytest.raises(RuntimeError):
        LBM.simular(mask, u_entrada=0.4, tau=0.5001, iteracoes=4000, intervalo=100, mostrar_progresso=False)
