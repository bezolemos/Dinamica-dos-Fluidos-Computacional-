"""Liga a foto e o túnel do Projeto 17 ao movimento do Projeto 15."""

from dataclasses import dataclass
from pathlib import Path
import math
import warnings

from aerodinamica import geometry, lbm, metrics
from . import movimento

FOTO_EXEMPLO = Path(__file__).resolve().parents[1] / "exemplos" / "carrinho.png"
LARGURA_MALHA, ALTURA_MALHA = 120, 48
U_MALHA, TAU, PASSOS_LBM, TOLERANCIA = 0.05, 0.8, 7000, 1e-4
DENSIDADE_AR = 1.225


@dataclass(frozen=True)
class DadosCarro:
    # Os valores iniciais são apenas uma configuração para experimentar.
    roda_cm: float = 10
    eixo_cm: float = 1
    corda_cm: float = 100
    haste_cm: float = 15
    massa_g: float = 200
    meta_m: float = 10
    k_mola: float = 0.2
    angulo_inicial: float = 90
    angulo_partida: float = 90
    atrito: float = 0.6
    rolamento: float = 0.02
    comprimento_cm: float = 25
    largura_cm: float = 8
    vento_ms: float = 1
    frente: str = "Esquerda"

    def parametros_movimento(self):
        return dict(
            diametro_roda_cm=self.roda_cm, diametro_eixo_cm=self.eixo_cm,
            comprimento_corda_cm=self.corda_cm, comprimento_haste_cm=self.haste_cm,
            massa_total_carrinho_g=self.massa_g, distancia_meta_m=self.meta_m,
            constante_torsional_mola=self.k_mola, angulo_inicial_da_mola_em_graus=self.angulo_inicial,
            angulo_atual_da_mola_em_graus=self.angulo_partida,
            coeficiente_atrito_estatico=self.atrito, coeficiente_resistencia_rolamento=self.rolamento,
        )


@dataclass
class Tunel:
    chave: tuple
    mascara: object
    rho: object
    ux: object
    uy: object
    metricas: dict
    avisos: tuple
    iteracoes: int
    convergiu: bool


@dataclass
class Ensaio:
    carro: DadosCarro
    foto: Path
    tunel: Tunel
    mecanico: dict
    ar_parado: dict
    com_vento: dict
    area_m2: float
    perda_m: float
    perda_pct: float | None
    velocidade_relativa_ms: float
    recomendacoes: list[str]


def simular_projetos(foto, carro, *, roi=None, tunel_anterior=None, progresso=lambda texto: None):
    """Só a foto, o sentido e o recorte determinam o campo de referência reutilizável."""
    for valor in (carro.comprimento_cm, carro.largura_cm):
        if not math.isfinite(valor) or valor <= 0:
            raise ValueError("O comprimento e a largura reais do carrinho precisam ser positivos.")
    if not math.isfinite(carro.vento_ms) or carro.vento_ms < 0:
        raise ValueError("O vento de frente deve ser pelo menos 0 m/s. Use zero para ar parado.")
    if carro.frente not in ("Esquerda", "Direita"):
        raise ValueError("Informe para qual lado a frente aponta na foto.")
    foto = Path(foto).resolve()
    if not foto.is_file():
        raise ValueError("Escolha uma foto do carrinho que esteja disponível neste computador.")
    arq = foto.stat()
    chave = (foto, arq.st_size, arq.st_mtime_ns, carro.frente, tuple(roi) if roi else None)
    params = carro.parametros_movimento()
    mecanico = movimento.executar_simulacao(**params)
    if tunel_anterior is not None and tunel_anterior.chave == chave:
        tunel = tunel_anterior
    else:
        progresso("Identificando a silhueta do carrinho…")
        with warnings.catch_warnings(record=True) as alertas:
            warnings.simplefilter("always", UserWarning)
            mascara = geometry.criar_mascara(str(foto), LARGURA_MALHA, ALTURA_MALHA,
                                              roi=roi, espelhar_horizontal=carro.frente == "Direita")
        avisos = [str(a.message) for a in alertas if issubclass(a.category, UserWarning)]
        # Um ponto ou um risco no lugar da carroceria não permite estimar uma área.
        if metrics.comprimento_carro(mascara) < 4 or metrics.area_frontal(mascara) < 2:
            raise ValueError("A silhueta ficou pequena demais. Recorte a foto ao redor do carro inteiro.")
        passo, variacao = 0, math.inf

        def informar(i, erro):
            nonlocal passo, variacao
            passo, variacao = i, erro
            progresso(f"Calculando o fluxo de ar · passo {i} de até {PASSOS_LBM}…")

        rho, ux, uy = lbm.simular(mascara, u_entrada=U_MALHA, tau=TAU, iteracoes=PASSOS_LBM,
                                  tolerancia=TOLERANCIA, mostrar_progresso=False, ao_verificar=informar)
        met = metrics.calcular_metricas(rho, ux, uy, mascara, U_MALHA, TAU)
        if not math.isfinite(met["coeficiente_arrasto"]) or met["coeficiente_arrasto"] <= 0:
            raise ValueError("O túnel não produziu arrasto utilizável. Confira a silhueta e repita o ensaio.")
        if variacao >= TOLERANCIA:
            avisos.append("O túnel chegou ao limite de passos antes da estabilização. Compare os valores com cautela.")
        tunel = Tunel(chave, mascara, rho, ux, uy, met, tuple(avisos + met["avisos"]),
                      passo, variacao < TOLERANCIA)
    progresso("Calculando a corrida em ar parado e com vento…")
    real = metrics.converter_para_real(tunel.metricas, carro.comprimento_cm / 100,
                                       carro.vento_ms, largura_m=carro.largura_cm / 100,
                                       rho_ar=DENSIDADE_AR)
    cd_area = tunel.metricas["coeficiente_arrasto"] * real["area"]
    ar = dict(cd_area_m2=cd_area, densidade_ar=DENSIDADE_AR)
    parado = movimento.executar_simulacao(**params, **ar)
    vento = (movimento.executar_simulacao(**params, **ar, vento_frontal_ms=carro.vento_ms)
             if carro.vento_ms else parado)
    d0 = parado["movimento"]["distancia_total_m"]
    perda = d0 - vento["movimento"]["distancia_total_m"]
    percentual = 100 * perda / d0 if d0 > 0 else None
    sugestoes = list(vento["recomendacoes"])
    if perda > 0.01:
        sugestoes.insert(0, "Teste um perfil mais baixo ou uma frente mais estreita. Repita o ensaio com o mesmo vento e compare os alcances.")
    if vento["movimento"]["distancia_total_m"] == 0:
        sugestoes.insert(0, "O carro ficou parado. Confira o ângulo de partida e compare a tração da mola com o rolamento e o vento.")
    if not sugestoes:
        sugestoes.append("Faça uma corrida com essas medidas e compare a distância medida com a estimativa.")
    v_rel = carro.vento_ms + vento["movimento"]["velocidade_maxima_m_s"]
    return Ensaio(carro, foto, tunel, mecanico, parado, vento, real["area"], perda, percentual, v_rel, sugestoes)
