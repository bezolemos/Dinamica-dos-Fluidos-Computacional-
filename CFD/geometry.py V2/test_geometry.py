"""
Testes do geometry.py.   Execute com:   pytest tests/ -v

Filosofia: cada teste compara a máscara com uma VERDADE conhecida (a forma
que foi desenhada), e não apenas "retornou alguma coisa". As imagens são
sintéticas; elas NÃO substituem testes com fotografias reais.
Limitações conhecidas estão marcadas com xfail(strict=True): se um dia
passarem, o teste avisa para atualizar a documentação.
"""
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry  # noqa: E402
from geometry import criar_mascara  # noqa: E402

LARG, ALT = 900, 500
OITO = np.ones((3, 3), dtype=bool)


# ------------------------------------------------------------------
# Utilitários: cena sintética com verdade conhecida
# ------------------------------------------------------------------

def _fundo(tipo, rng):
    yy, xx = np.mgrid[0:ALT, 0:LARG]
    cheio = lambda rgb: np.broadcast_to(np.array(rgb, float), (ALT, LARG, 3)).copy()
    if tipo == "branco":
        return cheio((255, 255, 255))
    if tipo == "preto":
        return cheio((0, 0, 0))
    if tipo == "cinza":
        return cheio((128, 128, 128))
    if tipo == "verde":
        return cheio((110, 160, 90))
    if tipo == "gradiente":
        return np.stack([150 + 80 * xx / LARG, 160 + 60 * yy / ALT, 170 + 0 * xx], -1)
    if tipo == "textura":
        r = rng.normal(0, 1, (ALT // 25 + 2, LARG // 25 + 2)).astype(np.float32)
        r = cv2.resize(r, (LARG, ALT), interpolation=cv2.INTER_CUBIC)
        return np.stack([200 + 12 * r, 200 + 12 * r, 205 + 12 * r], -1)
    if tipo == "parede_e_chao":          # dois fundos distintos, com borda reta
        f = cheio((215, 215, 215))
        f[360:, :] = (120, 115, 110)
        return f
    raise ValueError(tipo)


def cena(fundo="branco", corpo=(20, 20, 20), ruido=0.0, blur=0.0, asa=True,
         sombra=False, rodas_separadas=False, seed=0):
    """Devolve (imagem PIL RGB, verdade bool). O carrinho é um chassi fino,
    cabine, duas rodas escuras e (opcionalmente) uma asa de 4 px de espessura."""
    rng = np.random.default_rng(seed)
    im = Image.fromarray(_fundo(fundo, rng).clip(0, 255).astype(np.uint8))
    if sombra:   # escurecimento suave sob o carrinho (sombra física)
        sh = Image.new("L", (LARG, ALT), 0)
        ImageDraw.Draw(sh).ellipse([100, 395, 760, 440], fill=110)
        sh = sh.filter(ImageFilter.GaussianBlur(10))
        im = Image.composite(Image.new("RGB", (LARG, ALT), (0, 0, 0)), im, sh)

    verdade = Image.new("L", (LARG, ALT), 0)
    d, dv = ImageDraw.Draw(im), ImageDraw.Draw(verdade)

    def parte(fn, cor):
        fn(d, cor)
        fn(dv, 255)

    dy = 120 if rodas_separadas else 0      # rodas soltas do corpo (não tocam)
    parte(lambda g, c: g.rectangle([160, 310, 640, 335], fill=c), corpo)
    parte(lambda g, c: g.polygon([(300, 310), (340, 270), (420, 270), (450, 310)], fill=c), corpo)
    parte(lambda g, c: g.ellipse([90, 290 + dy, 190, 390 + dy], fill=c), (30, 30, 30))
    parte(lambda g, c: g.ellipse([560, 270 + dy, 700, 410 + dy], fill=c), (25, 25, 25))
    if asa:
        parte(lambda g, c: g.line([(560, 300), (780, 190)], fill=c, width=4), (50, 50, 55))

    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.array(im).astype(float)
    if ruido:
        a += rng.normal(0, ruido, a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)), np.array(verdade) > 0


def _recorte(m):
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def iou_forma(mask, verdade):
    """IoU entre a máscara e a verdade, ambas recortadas na própria caixa e
    comparadas na mesma resolução. Independe de posição/escala escolhidas."""
    a = _recorte(mask)
    v = _recorte(verdade)
    v = cv2.resize(v.astype(np.float32), (a.shape[1], a.shape[0]),
                   interpolation=cv2.INTER_AREA) >= 0.5
    return (a & v).sum() / max(1, (a | v).sum())


def componentes(m):
    return ndimage.label(m, structure=OITO)[1]


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


@pytest.fixture
def salvar(tmp_path):
    def _salvar(imagem, nome="img.png", **kw):
        caminho = tmp_path / nome
        imagem.save(caminho, **kw)
        return str(caminho)
    return _salvar


@pytest.fixture
def basica(salvar):
    im, v = cena()
    return salvar(im), v


# ------------------------------------------------------------------
# Contrato de saída
# ------------------------------------------------------------------

def test_saida_bool_com_shape_altura_largura(basica):
    caminho, _ = basica
    mask = criar_mascara(caminho, 400, 200)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (200, 400)          # (altura, largura)
    assert 0 < mask.sum() < mask.size


def test_forma_fiel_a_verdade(basica):
    caminho, v = basica
    assert iou_forma(criar_mascara(caminho, 400, 200), v) >= 0.95


def test_nao_salva_arquivo_sem_pedir(basica, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caminho, _ = basica
    criar_mascara(caminho, 100, 100)
    assert not (tmp_path / "mascara.png").exists()


def test_resultado_reprodutivel(basica):
    caminho, _ = basica
    a = criar_mascara(caminho, 400, 200)
    b = criar_mascara(caminho, 400, 200)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------
# Formatos e modos de imagem
# ------------------------------------------------------------------

@pytest.mark.parametrize("nome,kw", [("a.png", {}), ("a.jpg", {"quality": 85}), ("a.bmp", {})])
def test_formatos_de_arquivo(salvar, nome, kw):
    im, v = cena()
    mask = criar_mascara(salvar(im, nome, **kw), 400, 200)
    assert iou_forma(mask, v) >= 0.93


@pytest.mark.parametrize("nome,kw", [
    ("a.jfif", {"format": "JPEG"}), ("a.jpe", {"format": "JPEG"}), ("a.jpeg", {}),
    ("a.webp", {"lossless": True}), ("a.tif", {}), ("a.tiff", {"compression": "tiff_lzw"}),
    ("a.gif", {}), ("a.ppm", {}), ("a.ico", {"sizes": [(256, 256)]}), ("a.avif", {"quality": 95}),
])
def test_outros_formatos_de_imagem(salvar, nome, kw):
    im, v = cena()
    if nome.endswith(".ico"):
        im, v = im.resize((256, 256)), cv2.resize(v.astype(np.uint8), (256, 256)) > 0
    try:
        caminho = salvar(im, nome, **kw)
    except (KeyError, OSError, ValueError):
        pytest.skip(f"este Pillow não grava {nome}")
    assert iou_forma(criar_mascara(caminho, 400, 200), v) >= 0.85


def test_formato_vem_do_conteudo_e_nao_da_extensao(salvar):
    # um JPEG salvo com extensão errada (.png) ou sem extensão ainda abre
    im, v = cena()
    for nome in ("jpeg_disfarcado.png", "sem_extensao"):
        caminho = salvar(im, nome, format="JPEG", quality=90)
        assert iou_forma(criar_mascara(caminho, 400, 200), v) >= 0.93


def test_lista_de_extensoes_inclui_os_formatos_comuns():
    ext = geometry.extensoes_suportadas()
    for e in (".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".webp", ".tif", ".tiff", ".gif"):
        assert e in ext


def test_formato_desconhecido_da_erro_com_dica(tmp_path):
    f = tmp_path / "desenho.svg"
    f.write_text("<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>")
    with pytest.raises(ValueError, match="SVG"):
        criar_mascara(str(f), 100, 100)


@pytest.mark.parametrize("modo", ["L", "I;16"])
def test_escala_de_cinza_e_16_bits(salvar, modo):
    im, v = cena()
    cinza = im.convert("L")
    if modo == "I;16":
        cinza = Image.fromarray(np.array(cinza).astype(np.uint16) * 257)
    mask = criar_mascara(salvar(cinza), 400, 200)
    assert iou_forma(mask, v) >= 0.93


def test_orientacao_exif_e_respeitada(salvar):
    im, _ = cena()
    girada = im.rotate(90, expand=True)           # pixels em retrato
    exif = Image.Exif()
    exif[274] = 6                                  # "gire 90° horário para exibir"
    mask = criar_mascara(salvar(girada, "exif.jpg", exif=exif, quality=95), 400, 200)
    x0, y0, x1, y1 = bbox(mask)
    assert (x1 - x0 + 1) / (y1 - y0 + 1) > 1.5     # o carrinho é largo, não alto


def test_imagem_grande_e_reduzida_sem_mudar_a_forma(salvar):
    im, v = cena()
    grande = im.resize((LARG * 3, ALT * 3), Image.LANCZOS)
    mask = criar_mascara(salvar(grande, "g.png"), 400, 200, lado_maximo=600)
    assert iou_forma(mask, v) >= 0.93


# ------------------------------------------------------------------
# Fundos
# ------------------------------------------------------------------

def test_fundo_branco_e_preto_dao_mesma_geometria(salvar):
    # O desenho é o mesmo; só as cores trocam (negativo da imagem).
    im, v = cena("branco", corpo=(20, 20, 20))
    neg, _ = cena("preto", corpo=(235, 235, 235))
    a = criar_mascara(salvar(im, "a.png"), 400, 200)
    b = criar_mascara(salvar(neg, "b.png"), 400, 200)
    assert iou_forma(a, v) >= 0.95 and iou_forma(b, v) >= 0.95
    assert (a & b).sum() / (a | b).sum() >= 0.95


@pytest.mark.parametrize("fundo", ["cinza", "verde", "gradiente", "parede_e_chao"])
@pytest.mark.parametrize("corpo", [(190, 30, 30), (20, 20, 20), (235, 235, 235)])
def test_fundos_coloridos_e_nao_uniformes(salvar, fundo, corpo):
    im, v = cena(fundo, corpo=corpo)
    mask = criar_mascara(salvar(im), 400, 200, usar_grabcut=False)
    assert iou_forma(mask, v) >= 0.95


def test_fundo_com_textura_exige_grabcut(salvar):
    im, v = cena("textura", corpo=(190, 30, 30))
    assert iou_forma(criar_mascara(salvar(im), 400, 200, usar_grabcut=True), v) >= 0.9


@pytest.mark.parametrize("ruido,blur", [(6, 0), (0, 1.5), (6, 1.5)])
def test_ruido_e_desfoque(salvar, ruido, blur):
    im, v = cena("cinza", corpo=(150, 150, 155), ruido=ruido, blur=blur)
    assert iou_forma(criar_mascara(salvar(im), 400, 200), v) >= 0.85


def test_parte_clara_parecida_com_o_fundo_nao_some(salvar):
    # corpo cinza-claro (contraste baixo) ligando duas rodas escuras
    im, v = cena("branco", corpo=(205, 208, 212), asa=False)
    mask = criar_mascara(salvar(im), 400, 200)
    assert componentes(mask) == 1                   # o corpo liga as rodas
    assert iou_forma(mask, v) >= 0.9


def test_sensibilidade_maior_nao_diminui_a_mascara(salvar):
    im, _ = cena("cinza", corpo=(150, 150, 155), ruido=4)
    caminho = salvar(im)
    pouco = criar_mascara(caminho, 400, 200, sensibilidade=0.1, usar_grabcut=False).sum()
    muito = criar_mascara(caminho, 400, 200, sensibilidade=0.9, usar_grabcut=False).sum()
    assert muito >= pouco


# ------------------------------------------------------------------
# Transparência
# ------------------------------------------------------------------

def _rgba(verdade, cor_carro, cor_fundo):
    arr = np.zeros(verdade.shape + (4,), np.uint8)
    arr[...] = (*cor_fundo, 0)
    arr[verdade] = (*cor_carro, 255)
    return Image.fromarray(arr, "RGBA")


def test_alpha_e_o_que_define_a_silhueta(salvar):
    # Carrinho e fundo com EXATAMENTE a mesma cor RGB: só o alfa os distingue.
    # Se o código ignorasse o canal alfa, daria erro de "cor única".
    _, v = cena()
    mask = criar_mascara(salvar(_rgba(v, (128, 128, 128), (128, 128, 128))), 400, 200)
    assert iou_forma(mask, v) >= 0.95


def test_png_com_grande_area_transparente(salvar):
    # carrinho minúsculo (~0,1% da tela) em uma tela de 4000x4000
    _, v = cena()
    pequeno = cv2.resize(_recorte(v).astype(np.uint8), (160, 52),
                         interpolation=cv2.INTER_NEAREST) > 0
    tela = np.zeros((4000, 4000), bool)
    tela[1800:1800 + 52, 1500:1500 + 160] = pequeno
    mask = criar_mascara(salvar(_rgba(tela, (255, 255, 255), (0, 0, 0)), "tela.png"), 400, 200)
    x0, y0, x1, y1 = bbox(mask)
    assert abs((x1 - x0 + 1) - 160) <= 2            # ocupa 40% da largura da malha
    assert iou_forma(mask, tela) >= 0.9


def test_png_totalmente_transparente_da_erro(salvar):
    caminho = salvar(Image.new("RGBA", (300, 300), (0, 0, 0, 0)))
    with pytest.raises(ValueError, match="transparente"):
        criar_mascara(caminho, 100, 100)


def test_png_rgba_opaco_usa_segmentacao_por_cor(salvar):
    im, v = cena()
    mask = criar_mascara(salvar(im.convert("RGBA")), 400, 200)
    assert iou_forma(mask, v) >= 0.95


# ------------------------------------------------------------------
# Geometria: peças finas, partes desconectadas, bordas, buracos
# ------------------------------------------------------------------

def test_asa_fina_nao_se_fragmenta_nem_some(basica):
    caminho, v = basica
    mask = criar_mascara(caminho, 400, 200)
    assert componentes(mask) == 1
    # a asa define o canto superior direito: a caixa deve bater com a verdade
    x0, y0, x1, y1 = bbox(mask)
    xv0, yv0, xv1, yv1 = bbox(v)
    esc = (x1 - x0 + 1) / (xv1 - xv0 + 1)
    assert abs((y1 - y0 + 1) - (yv1 - yv0 + 1) * esc) <= 2


def test_partes_desconectadas_sao_preservadas(salvar):
    im, _ = cena(rodas_separadas=True, asa=False)
    caminho = salvar(im)
    padrao = criar_mascara(caminho, 400, 200)
    so_maior = criar_mascara(caminho, 400, 200, manter_maior_objeto=True)
    assert componentes(padrao) == 3                  # corpo + 2 rodas
    assert componentes(so_maior) == 1
    assert padrao.sum() > so_maior.sum()


def test_sujeira_pequena_e_descartada(salvar):
    im, v = cena()
    ImageDraw.Draw(im).ellipse([30, 30, 36, 36], fill=(20, 20, 20))   # pontinho no fundo
    mask = criar_mascara(salvar(im), 400, 200)
    assert componentes(mask) == 1
    assert iou_forma(mask, v) >= 0.95


def test_carrinho_encostado_na_borda_da_imagem(salvar):
    im = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(im).rectangle([0, 200, 500, 399], fill="black")     # 501x200
    mask = criar_mascara(salvar(im), 400, 200)
    r = _recorte(mask)
    assert r.mean() >= 0.97                                           # retângulo cheio
    assert abs(r.shape[1] / r.shape[0] - 501 / 200) <= 0.15 * 501 / 200


def test_vao_aberto_continua_fluido(salvar):
    # "ponte": o vão embaixo está ABERTO para o exterior e deve continuar ar
    im = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([100, 150, 800, 350], fill="black")
    d.rectangle([300, 250, 600, 350], fill="white")     # vão aberto por baixo
    mask = criar_mascara(salvar(im), 400, 200, fracao_dominio=0.8)
    x0, y0, x1, y1 = bbox(mask)
    assert not mask[y1 - 3, (x0 + x1) // 2]


def test_bolsao_totalmente_fechado_e_preenchido(salvar):
    # um furo FECHADO não é alcançado pelo escoamento externo; vira sólido
    im = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([100, 150, 800, 350], fill="black")
    d.rectangle([400, 220, 500, 280], fill="white")     # janela fechada
    mask = criar_mascara(salvar(im), 400, 200, fracao_dominio=0.8)
    x0, y0, x1, y1 = bbox(mask)
    assert mask[(y0 + y1) // 2, (x0 + x1) // 2]


def test_nenhum_bolsao_de_fluido_isolado(basica):
    caminho, _ = basica
    mask = criar_mascara(caminho, 400, 200)
    assert componentes(~mask) == 1                    # todo o ar é uma região só


def test_vao_so_diagonal_nao_e_preenchido():
    # No D2Q9 o fluido atravessa uma fresta diagonal; ela não pode ser selada.
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    m[12:28, 12:28] = False                # interior vazio
    m[10, 10] = False                      # canto aberto: ligação só pela diagonal
    m[10, 11] = m[11, 10] = True
    m[11, 11] = False
    saida = geometry._reduzir_preservando_pecas_finas(m, 40, 40)
    assert not saida[20, 20]               # interior continua fluido


# ------------------------------------------------------------------
# Domínio computacional
# ------------------------------------------------------------------

@pytest.mark.parametrize("fracao,px,py", [(0.4, 0.35, 0.5), (0.25, 0.2, 0.3), (0.6, 0.5, 0.5), (0.3, 0.8, 0.7)])
def test_tamanho_e_posicao_no_dominio(basica, fracao, px, py):
    caminho, v = basica
    L, A = 400, 200
    mask = criar_mascara(caminho, L, A, fracao_dominio=fracao, posicao_x=px, posicao_y=py)
    x0, y0, x1, y1 = bbox(mask)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    # a dimensão limitante ocupa a fração pedida (±1,5 célula)
    assert min(abs(w - L * fracao), abs(h - A * fracao)) <= 1.5
    # centro na posição pedida (±1,5 célula), pois cabe no domínio
    assert abs((x0 + x1 + 1) / 2 - px * L) <= 1.5
    assert abs((y0 + y1 + 1) / 2 - py * A) <= 1.5
    # sem cortes: a proporção da caixa é a do desenho
    xv0, yv0, xv1, yv1 = bbox(v)
    assert abs(w / h - (xv1 - xv0 + 1) / (yv1 - yv0 + 1)) <= 0.08 * w / h


@pytest.mark.parametrize("px,py", [(0, 0), (1, 1), (0, 1), (1, 0)])
def test_posicoes_extremas_nao_cortam_o_carrinho(basica, px, py):
    caminho, _ = basica
    referencia = criar_mascara(caminho, 400, 200, posicao_x=0.5, posicao_y=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extrema = criar_mascara(caminho, 400, 200, posicao_x=px, posicao_y=py)
    assert extrema.sum() == referencia.sum()          # nenhuma célula perdida
    x0, y0, x1, y1 = bbox(extrema)
    assert x0 >= 0 and y0 >= 0 and x1 < 400 and y1 < 200


def test_carrinho_nao_toca_as_bordas_com_configuracao_padrao(basica):
    caminho, _ = basica
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # nem aviso deve sair
        mask = criar_mascara(caminho, 400, 200)
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def test_avisa_quando_toca_a_borda_do_dominio(basica):
    caminho, _ = basica
    with pytest.warns(UserWarning, match="borda"):
        criar_mascara(caminho, 400, 200, fracao_dominio=1.0)


def test_espelhar_horizontal(basica):
    caminho, _ = basica
    normal = criar_mascara(caminho, 400, 200, posicao_x=0.5)
    espelhada = criar_mascara(caminho, 400, 200, posicao_x=0.5, espelhar_horizontal=True)
    esperada = normal[:, ::-1]
    assert not np.array_equal(normal, espelhada)       # o carrinho é assimétrico
    assert (espelhada & esperada).sum() / (espelhada | esperada).sum() >= 0.95


@pytest.mark.parametrize("largura,altura,minimo", [(800, 400, 0.93), (400, 200, 0.93), (200, 100, 0.88)])
def test_varias_resolucoes(basica, largura, altura, minimo):
    caminho, v = basica
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # resolução razoável: sem avisos
        mask = criar_mascara(caminho, largura, altura)
    assert mask.shape == (altura, largura) and mask.dtype == bool
    assert iou_forma(mask, v) >= minimo
    assert componentes(mask) == 1


def test_resolucao_baixa_avisa_que_a_asa_fina_se_fragmentou(basica):
    # 100x50: o carrinho tem ~40 células e a asa (4 px) fica com ~0,2 célula
    caminho, v = basica
    with pytest.warns(UserWarning, match="partes"):
        mask = criar_mascara(caminho, 100, 50)
    assert iou_forma(mask, v) >= 0.75                 # o corpo continua correto


def test_resolucao_minima_funciona_e_avisa(basica):
    caminho, _ = basica
    with pytest.warns(UserWarning):
        mask = criar_mascara(caminho, 1, 1)
    assert mask.shape == (1, 1) and mask[0, 0]


def test_area_na_malha_proxima_da_esperada(basica):
    # o redimensionamento não pode inflar/encolher a geometria de forma relevante
    caminho, v = basica
    mask = criar_mascara(caminho, 400, 200)
    xv0, yv0, xv1, yv1 = bbox(v)
    esc = (bbox(mask)[2] - bbox(mask)[0] + 1) / (xv1 - xv0 + 1)
    assert 0.95 <= mask.sum() / (v.sum() * esc * esc) <= 1.06


# ------------------------------------------------------------------
# Validação de entradas e erros
# ------------------------------------------------------------------

@pytest.mark.parametrize("largura,altura", [(0, 100), (-5, 100), (100, 0), (100.5, 100), ("100", 100), (True, 100), (None, 100)])
def test_dimensoes_invalidas(basica, largura, altura):
    with pytest.raises(ValueError):
        criar_mascara(basica[0], largura, altura)


@pytest.mark.parametrize("kw", [
    {"lado_maximo": 0}, {"lado_maximo": 10.5},
    {"fracao_dominio": 0}, {"fracao_dominio": 1.1}, {"fracao_dominio": float("nan")}, {"fracao_dominio": "0.4"},
    {"posicao_x": -0.1}, {"posicao_y": 1.5}, {"sensibilidade": 0},
    {"area_minima_relativa": 1.0}, {"ocupacao_maxima": 0}, {"ocupacao_maxima": 2},
])
def test_parametros_invalidos(basica, kw):
    with pytest.raises(ValueError):
        criar_mascara(basica[0], 100, 100, **kw)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrada"):
        criar_mascara(str(tmp_path / "nao_existe.png"), 100, 100)


def test_diretorio_no_lugar_de_arquivo(tmp_path):
    with pytest.raises(FileNotFoundError):
        criar_mascara(str(tmp_path), 100, 100)


def test_arquivo_que_nao_e_imagem(tmp_path):
    falso = tmp_path / "texto.png"
    falso.write_text("isto não é uma imagem")
    with pytest.raises(ValueError, match="abrir"):
        criar_mascara(str(falso), 100, 100)


def test_imagem_truncada(tmp_path):
    im, _ = cena()
    caminho = tmp_path / "t.jpg"
    im.save(caminho, quality=90)
    dados = caminho.read_bytes()
    caminho.write_bytes(dados[: len(dados) // 3])
    with pytest.raises(ValueError, match="abrir"):
        criar_mascara(str(caminho), 100, 100)


def test_imagem_de_cor_unica(salvar):
    with pytest.raises(ValueError, match="cor única"):
        criar_mascara(salvar(Image.new("RGB", (200, 200), "white")), 100, 100)


def test_fundo_segmentado_como_carrinho_e_rejeitado(salvar):
    # objeto cobrindo ~89% da imagem: quase certamente é um erro de segmentação
    im = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(im).rectangle([20, 20, 779, 579], fill="black")
    caminho = salvar(im)
    with pytest.raises(ValueError, match="cobre"):
        criar_mascara(caminho, 100, 100)
    assert criar_mascara(caminho, 100, 100, ocupacao_maxima=None).any()   # opt-out explícito


def test_pathlib_e_nome_com_acentos(salvar):
    im, v = cena()
    caminho = salvar(im, "ção do carro.png")
    assert iou_forma(criar_mascara(Path(caminho), 400, 200), v) >= 0.95


# ------------------------------------------------------------------
# Saída opcional (PNG) e modo de depuração
# ------------------------------------------------------------------

def test_salvar_png_ida_e_volta(basica, tmp_path):
    caminho, _ = basica
    saida = tmp_path / "saida.png"
    mask = criar_mascara(caminho, 400, 200, salvar_em=str(saida))
    lida = np.array(Image.open(saida)) > 0
    assert lida.shape == mask.shape and np.array_equal(lida, mask)


def test_salvar_em_pasta_inexistente_levanta_erro(basica, tmp_path):
    with pytest.raises(OSError, match="salvar"):
        criar_mascara(basica[0], 100, 100, salvar_em=str(tmp_path / "nao" / "existe" / "x.png"))


def test_debug_salva_todas_as_etapas(basica, tmp_path):
    caminho, _ = basica
    pasta = tmp_path / "dbg"
    criar_mascara(caminho, 400, 200, debug_dir=str(pasta))
    esperados = {"0_entrada.png", "1_forte.png", "2_fraca.png", "3_grabcut.png",
                 "4_limpa.png", "5_sobreposicao.png", "6_final_na_malha.png"}
    assert esperados <= set(os.listdir(pasta))
    entrada = np.array(Image.open(pasta / "0_entrada.png"))
    sobre = np.array(Image.open(pasta / "5_sobreposicao.png"))
    assert entrada.shape == sobre.shape
    assert (sobre[..., 0].astype(int) - sobre[..., 2]).max() > 100     # vermelho sobre o carrinho
    assert np.array(Image.open(pasta / "6_final_na_malha.png")).shape == (200, 400)


# ------------------------------------------------------------------
# Seletor de arquivos (explorador)
# ------------------------------------------------------------------

def _tk_falso(monkeypatch, retorno):
    """Instala um 'tkinter' falso cujo diálogo devolve `retorno`."""
    import types
    tk = types.ModuleType("tkinter")
    fd = types.ModuleType("tkinter.filedialog")
    eventos = []

    class Raiz:
        def withdraw(self): eventos.append("withdraw")
        def attributes(self, *a): pass
        def destroy(self): eventos.append("destroy")

    tk.Tk, tk.TclError, tk.filedialog = Raiz, Exception, fd
    fd.askopenfilename = lambda **kw: retorno
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fd)
    return eventos


def test_explorador_devolve_o_caminho_escolhido(monkeypatch):
    eventos = _tk_falso(monkeypatch, "/algum/carrinho.png")
    assert geometry.escolher_imagem() == "/algum/carrinho.png"
    assert "destroy" in eventos                       # a janela é sempre fechada


def test_explorador_cancelado_devolve_none(monkeypatch):
    _tk_falso(monkeypatch, "")
    assert geometry.escolher_imagem() is None


def test_sem_interface_grafica_pede_o_caminho_no_terminal(monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)     # import falha
    monkeypatch.setattr("builtins.input", lambda *_: ' "C:\\imgs\\carro.png" ')
    assert geometry.escolher_imagem() == "C:\\imgs\\carro.png"


# ------------------------------------------------------------------
# LIMITAÇÕES CONHECIDAS (xfail estrito: quando passarem, atualize a doc)
# ------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Sombra no chão só mais escura que o fundo é "
                   "indistinguível de uma peça escura; medido IoU ~0,4.")
def test_sombra_suave_nao_deveria_entrar_na_mascara(salvar):
    im, v = cena("cinza", corpo=(150, 150, 155), sombra=True)
    assert iou_forma(criar_mascara(salvar(im), 400, 200), v) >= 0.85


@pytest.mark.xfail(strict=True, reason="Corpo de baixo contraste sobre fundo com textura "
                   "forte: ruído do fundo e corpo ficam na mesma faixa; medido IoU ~0,36.")
def test_corpo_claro_em_fundo_com_textura(salvar):
    im, v = cena("textura", corpo=(190, 195, 200))
    assert iou_forma(criar_mascara(salvar(im), 400, 200), v) >= 0.85
