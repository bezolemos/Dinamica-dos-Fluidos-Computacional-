"""
geometry.py - Projeto 17 (Simulador aerodinâmico 2D com LBM D2Q9)

Transforma a silhueta 2D de um carrinho (imagem) em uma máscara binária
que o solver consegue usar.

Uso rápido
----------
    python geometry.py                 # abre o explorador de arquivos
    python geometry.py imagem.png      # usa o caminho informado

Contrato com o restante do projeto
----------------------------------
Entrada : caminho da imagem, largura e altura da malha (em células)
Saída   : `mask`, np.ndarray de bool com shape (altura, largura)
              True  -> célula SÓLIDA (carrinho)
              False -> célula de FLUIDO (ar)

Convenção de eixos (a confirmar com o solver_lbm.py):
    mask[j, i]: j = linha = eixo y, com j = 0 no TOPO da imagem;
                i = coluna = eixo x, crescendo para a direita.
    O carrinho é colocado como aparece na imagem. Se ele estiver virado
    para a direita e o vento entrar pela esquerda, o vento bate na traseira:
    use espelhar_horizontal=True nesse caso.
"""

import argparse
import importlib
import json
import numbers
import os
import threading
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


class _ModuloPreguicoso:
    """
    Importa o módulo só no primeiro uso. OpenCV e SciPy demoram vários
    segundos para carregar (principalmente na primeira vez); assim a janela
    de escolha de arquivo abre na hora e eles carregam depois.
    """

    def __init__(self, nome):
        self.__dict__["_nome"] = nome
        self.__dict__["_mod"] = None

    def _carregar(self):
        if self.__dict__["_mod"] is None:
            self.__dict__["_mod"] = importlib.import_module(self.__dict__["_nome"])
        return self.__dict__["_mod"]

    def __getattr__(self, atributo):
        return getattr(self._carregar(), atributo)


cv2 = _ModuloPreguicoso("cv2")
ndimage = _ModuloPreguicoso("scipy.ndimage")
_spatial = _ModuloPreguicoso("scipy.spatial")


def _pre_carregar_bibliotecas():
    """Carrega as bibliotecas pesadas em segundo plano, enquanto o usuário escolhe o arquivo."""
    def tarefa():
        for modulo in (cv2, ndimage, _spatial):
            try:
                modulo._carregar()
            except Exception:
                pass
    threading.Thread(target=tarefa, daemon=True).start()


_8_VIZINHOS = np.ones((3, 3), dtype=bool)
_heif_tentado = False


def _registrar_heif():
    """Registra o leitor de HEIC/HEIF (fotos de iPhone), se o pacote opcional existir."""
    global _heif_tentado
    if _heif_tentado:
        return False
    _heif_tentado = True
    try:
        import pillow_heif  # type: ignore  (opcional; evita o aviso do Pylance)
        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        return False


def extensoes_suportadas():
    """
    Extensões de imagem que podem ser abertas (png, jpg, jfif, webp, tiff,
    gif, bmp, avif, ico, ppm...). O Pillow identifica o formato pelo CONTEÚDO
    do arquivo, então mesmo uma extensão incomum costuma funcionar.
    """
    Image.init()
    # formatos que o Pillow só grava, ou que não são imagens abríveis aqui
    ignorar = {".pdf", ".h5", ".hdf", ".grib", ".bufr", ".mpeg", ".mpg",
               ".eps", ".ps", ".emf", ".wmf"}
    return sorted(e for e in Image.registered_extensions() if e not in ignorar)


# -------------------------------------------------------------------
# Entrada / saída
# -------------------------------------------------------------------

def _para_8bits(imagem):
    """Converte imagens de 16/32 bits ou float para 8 bits (L) sem estourar."""
    arr = np.asarray(imagem).astype(np.float32)
    if imagem.mode.startswith("I;16"):
        arr = arr / 257.0
    else:
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "L")


def _carregar_imagem(caminho_imagem, lado_maximo):
    """
    Abre a imagem e devolve (rgb, alpha).

    - rgb   : uint8 (H, W, 3). Pixels transparentes viram branco.
    - alpha : uint8 (H, W) se a imagem tiver transparência REAL (há pixels
              transparentes e opacos); caso contrário None.
    - Com alpha, a imagem é recortada na caixa do objeto ANTES de reduzir
      (um carrinho pequeno numa tela enorme não pode sumir na redução).
    - Imagens maiores que `lado_maximo` são reduzidas.
    """
    try:
        imagem = Image.open(caminho_imagem)
    except UnidentifiedImageError:
        if not _registrar_heif():      # talvez seja HEIC/HEIF (pacote opcional)
            raise
        imagem = Image.open(caminho_imagem)
    imagem = ImageOps.exif_transpose(imagem)  # respeita a rotação do celular

    if imagem.mode in ("I", "F") or imagem.mode.startswith("I;16"):
        imagem = _para_8bits(imagem)

    tem_canal_alpha = (
        imagem.mode in ("RGBA", "LA", "PA")
        or (imagem.mode == "P" and "transparency" in imagem.info)
    )

    alpha = None
    if tem_canal_alpha:
        rgba = imagem.convert("RGBA")
        a = np.array(rgba.getchannel("A"))
        transparente = a < 128
        opaco = ~transparente
        if not opaco.any():
            raise ValueError("A imagem é totalmente transparente: não há carrinho.")
        if transparente.mean() > 0.001 and opaco.sum() >= 25:
            alpha = a
            ys, xs = np.nonzero(opaco)
            m = 2  # pequena folga
            y0, y1 = max(0, ys.min() - m), ys.max() + 1 + m
            x0, x1 = max(0, xs.min() - m), xs.max() + 1 + m
            rgba = rgba.crop((x0, y0, x1, y1))
            alpha = alpha[y0:y1, x0:x1]
        # Sem isso, pixels transparentes (RGB geralmente preto) seriam
        # confundidos com um carrinho preto.
        fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        imagem = Image.alpha_composite(fundo, rgba)

    rgb = np.array(imagem.convert("RGB"))

    maior_lado = max(rgb.shape[:2])
    if maior_lado > lado_maximo:
        fator = lado_maximo / maior_lado
        novo = (max(1, round(rgb.shape[1] * fator)),
                max(1, round(rgb.shape[0] * fator)))
        rgb = cv2.resize(rgb, novo, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, novo, interpolation=cv2.INTER_AREA)

    return rgb, alpha


def _salvar_png(caminho, array):
    """Salva uma matriz (bool/uint8, 2D ou RGB) como PNG; levanta OSError se falhar."""
    a = np.asarray(array)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    try:
        Image.fromarray(a.astype(np.uint8)).save(caminho, format="PNG")
    except (OSError, ValueError) as erro:
        raise OSError(f"Não foi possível salvar {caminho!r}: {erro}") from erro
    if not os.path.isfile(caminho):
        raise OSError(f"O arquivo {caminho!r} não foi criado.")


# -------------------------------------------------------------------
# Segmentação (carrinho x fundo)
# -------------------------------------------------------------------

def _centros_de_fundo(pixels_borda):
    """
    Estima UMA ou mais cores de fundo a partir das bordas da imagem.

    Fundos com duas regiões (parede + chão, céu + grama) têm mais de uma
    cor de fundo. Usa k-means (k=3) e mantém só centros que representam
    >= 10% da borda e estão afastados entre si (>= 25 em LAB). Assim um
    carrinho que encosta numa pequena parte da borda não vira "fundo".
    """
    pontos = pixels_borda.astype(np.float32)
    if len(pontos) > 20000:
        pontos = pontos[:: len(pontos) // 20000 + 1]

    centros = [np.median(pontos, axis=0)]
    if len(pontos) >= 100:
        criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(12345)  # resultado reprodutível entre execuções
        _, rotulos, cs = cv2.kmeans(pontos, 3, None, criterio, 3,
                                    cv2.KMEANS_PP_CENTERS)
        pesos = np.bincount(rotulos.ravel(), minlength=3) / len(pontos)
        escolhidos = []
        for idx in np.argsort(-pesos):
            if pesos[idx] < 0.10:
                continue
            c = cs[idx]
            if all(np.linalg.norm(c - e) >= 25 for e in escolhidos):
                escolhidos.append(c)
        if escolhidos:
            centros = escolhidos
    return centros


def _modelos_de_fundo(lab, centros, excluir):
    """
    Para cada cor de fundo, ajusta uma superfície suave (polinômio de grau 2
    em x, y) usando apenas pixels não excluídos mais próximos daquela cor.
    Absorve gradientes de iluminação. Cada modelo é limitado ao intervalo de
    cores realmente observado (evita extrapolações absurdas).
    Devolve a lista de imagens (H, W, 3), uma por cor de fundo.
    """
    altura, largura = lab.shape[:2]
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    x = x / max(1, largura - 1) * 2 - 1
    y = y / max(1, altura - 1) * 2 - 1
    base = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    passo = max(1, max(altura, largura) // 200)
    ok = ~excluir[::passo, ::passo]
    amostra_lab = lab[::passo, ::passo][ok]
    amostra_base = base[::passo, ::passo][ok]

    if len(amostra_lab):
        d = np.stack([np.linalg.norm(amostra_lab - c, axis=1) for c in centros])
        rotulo = d.argmin(axis=0)
    else:
        rotulo = np.zeros(0, dtype=int)

    modelos = []
    for k, c in enumerate(centros):
        sel = rotulo == k
        if sel.sum() < 50:
            modelos.append(np.broadcast_to(c, lab.shape).copy())
            continue
        coef, *_ = np.linalg.lstsq(amostra_base[sel], amostra_lab[sel], rcond=None)
        lo = amostra_lab[sel].min(axis=0)
        hi = amostra_lab[sel].max(axis=0)
        modelos.append(np.clip(base @ coef, lo, hi))
    return modelos


def _distancia_ao_fundo(lab, modelos):
    """Menor distância (LAB) de cada pixel a qualquer modelo de fundo."""
    return np.stack([np.linalg.norm(lab - m, axis=2) for m in modelos]).min(axis=0)


def _mascara_por_cor(rgb, sensibilidade):
    """
    Segmenta o carrinho comparando cada pixel com o fundo (espaço LAB).

    1. Acha as cores de fundo nas bordas (uma ou mais) e uma máscara grosseira.
    2. Reajusta o fundo como superfície suave, ignorando o carrinho
       (gradientes de iluminação não viram "carrinho").
    3. Mede o ruído do fundo e define os limiares a partir dele.

    Devolve (forte, fraca):
      - forte : diferença grande (certamente carrinho)
      - fraca : diferença pequena, mas acima do ruído (provavelmente carrinho;
                pega partes claras que o Otsu deixaria de fora)
    """
    altura, largura = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    margem = max(5, int(min(altura, largura) * 0.03))
    margem = max(1, min(margem, altura // 2, largura // 2))
    pixels_borda = np.concatenate([
        lab[:margem, :].reshape(-1, 3),
        lab[-margem:, :].reshape(-1, 3),
        lab[:, :margem].reshape(-1, 3),
        lab[:, -margem:].reshape(-1, 3),
    ])
    centros = _centros_de_fundo(pixels_borda)

    # --- passo 1: estimativa grosseira (fundo constante) -----------------
    distancia = _distancia_ao_fundo(lab, [np.broadcast_to(c, lab.shape) for c in centros])
    dist_borda = np.min(
        [np.linalg.norm(pixels_borda - c, axis=1) for c in centros], axis=0)
    # mediana (e não percentil alto): o carrinho pode encostar na borda
    piso0 = 3.0 * np.median(dist_borda) + 4.0
    otsu0, _ = cv2.threshold(
        np.clip(distancia, 0, 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grosseira = distancia >= max(0.25 * otsu0, piso0)

    # --- passo 2: fundo suave, ignorando o carrinho ------------------------
    kernel = np.ones((15, 15), np.uint8)
    excluir = cv2.dilate(grosseira.astype(np.uint8), kernel) > 0
    for _ in range(2):
        modelos = _modelos_de_fundo(lab, centros, excluir)
        distancia = _distancia_ao_fundo(lab, modelos)
        referencia = distancia[~excluir] if (~excluir).any() else dist_borda
        piso = max(3.0 * np.median(referencia) + 1.0, 4.0)
        excluir = cv2.dilate((distancia >= piso).astype(np.uint8), kernel) > 0

    referencia = distancia[~excluir] if (~excluir).any() else dist_borda
    piso = max(3.0 * np.median(referencia) + 1.0, 4.0)

    distancia_u8 = np.clip(distancia, 0, 255).astype(np.uint8)
    if distancia_u8.max() < max(10, piso):
        raise ValueError(
            "Não foi possível distinguir o carrinho do fundo: a imagem parece ter uma "
            "cor única, ou o carrinho ocupa boa parte das bordas. Deixe um pouco de "
            "fundo ao redor dele ou aponte o carrinho (seleção guiada, roi=...).")

    limiar_otsu, _ = cv2.threshold(
        distancia_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    forte = distancia >= max(limiar_otsu, 2 * piso)
    fraca = distancia >= piso * (1.6 - 0.6 * sensibilidade)
    return forte, fraca


def _fechar(mask):
    """Fechamento morfológico com kernel de ~1% da imagem (ímpar, >= 3)."""
    k = max(3, int(round(min(mask.shape) * 0.01)) // 2 * 2 + 1)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8)) > 0


def _refinar_com_grabcut(rgb, forte, fraca, iteracoes=5):
    """
    Refina a segmentação com GrabCut (ajuda em fundos com textura).

    - As máscaras são fechadas antes (iniciação menos esburacada).
    - `forte` é sempre mantida: o GrabCut só decide a região intermediária.
      (Sem isso, ele apagava partes de contraste baixo do carrinho.)
    - Se algo falhar ou o resultado for suspeito, devolve `fraca`.
    """
    forte = _fechar(forte)
    fraca_f = _fechar(fraca) | forte
    altura, largura = fraca_f.shape

    nucleo = cv2.erode(forte.astype(np.uint8), np.ones((3, 3), np.uint8))
    vizinhanca = cv2.dilate(fraca_f.astype(np.uint8), np.ones((15, 15), np.uint8))

    gc = np.full((altura, largura), cv2.GC_BGD, dtype=np.uint8)
    gc[vizinhanca > 0] = cv2.GC_PR_BGD
    gc[fraca_f] = cv2.GC_PR_FGD
    gc[nucleo > 0] = cv2.GC_FGD

    try:
        cv2.setRNGSeed(12345)  # resultado reprodutível
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None,
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    iteracoes, cv2.GC_INIT_WITH_MASK)
    except cv2.error as erro:
        warnings.warn(f"GrabCut falhou ({erro}); usando a máscara por limiar.")
        return fraca

    resultado = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    razao = resultado.sum() / max(1, fraca_f.sum())
    if razao < 0.5 or razao > 3.0:
        warnings.warn("GrabCut deu um resultado suspeito; usando a máscara por limiar.")
        return fraca
    return resultado | forte


def _limpar_mascara(mask, manter_maior_objeto, area_minima_relativa):
    """
    Fecha pequenas falhas, descarta sujeira e preenche buracos fechados.

    - Sem abertura morfológica (apagaria eixos e asas finos).
    - Por padrão descarta só componentes pequenos em relação ao maior
      (rodas e asas separadas são preservadas).
    - Buracos só são preenchidos se estiverem fechados também na diagonal
      (conectividade 8): o que o D2Q9 consegue alcançar continua fluido.
      Bolsões totalmente fechados não participam do escoamento externo.
    """
    mask = _fechar(mask).astype(np.uint8)

    n, rotulos, estat, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = estat[1:, cv2.CC_STAT_AREA]
        if manter_maior_objeto:
            mask = rotulos == 1 + int(np.argmax(areas))
        else:
            minimo = max(3, area_minima_relativa * areas.max())
            mask = np.isin(rotulos, [i + 1 for i, a in enumerate(areas) if a >= minimo])

    return ndimage.binary_fill_holes(mask > 0, structure=_8_VIZINHOS)


# -------------------------------------------------------------------
# Segmentação GUIADA (fundos com muita coisa): caixa + pincéis
# -------------------------------------------------------------------
#
# Quando o fundo tem muitos objetos, estimar "a cor do fundo" pelas bordas
# não funciona. Nesse caso o usuário aponta onde está o carrinho:
#   roi    = (x0, y0, x1, y1)  caixa em volta do carrinho, em FRAÇÕES (0 a 1)
#            da imagem já com a rotação EXIF corrigida (independe da resolução)
#   marcas = {"carrinho": [traco, ...], "fundo": [traco, ...], "raio": 0.008}
#            traco = lista de pontos (x, y) em frações; raio = fração do maior
#            lado da imagem (espessura do pincel)
# Tudo é serializável em JSON, então a seleção pode ser salva e reutilizada.

def _caixa_em_pixels(roi, altura, largura):
    """Converte a caixa em frações para pixels, validando-a."""
    try:
        x0, y0, x1, y1 = (float(v) for v in roi)
    except (TypeError, ValueError):
        raise ValueError("roi deve ser (x0, y0, x1, y1) em frações de 0 a 1.") from None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"roi inválida {roi!r}: use 0 <= x0 < x1 <= 1 e 0 <= y0 < y1 <= 1.")
    px = (int(np.floor(x0 * largura)), int(np.floor(y0 * altura)),
          int(np.ceil(x1 * largura)), int(np.ceil(y1 * altura)))
    if px[2] - px[0] < 8 or px[3] - px[1] < 8:
        raise ValueError("A caixa (roi) é pequena demais; desenhe-a em volta do carrinho inteiro.")
    return px


def _rasterizar_marcas(tracos, altura, largura, raio):
    """Desenha os traços (frações) em uma máscara uint8 do tamanho da imagem."""
    m = np.zeros((altura, largura), np.uint8)
    espessura = max(1, int(round(2 * raio * max(altura, largura))))
    try:
        for traco in tracos or []:
            pts = np.array([[x * (largura - 1), y * (altura - 1)] for x, y in traco])
            pts = np.round(pts).astype(np.int32)
            if len(pts) == 1:
                cv2.circle(m, tuple(int(v) for v in pts[0]), espessura // 2, 255, -1)
            elif len(pts) > 1:
                cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, espessura)
    except (TypeError, ValueError):
        raise ValueError("marcas inválidas: cada traço deve ser uma lista de pontos (x, y).") from None
    return m > 0


def _validar_marcas(marcas):
    if marcas is None:
        return None
    if not isinstance(marcas, dict):
        raise ValueError("marcas deve ser um dict com 'carrinho', 'fundo' e 'raio'.")
    raio = _numero(marcas.get("raio", 0.008), "marcas['raio']", 0, 0.5, True)
    return {"carrinho": marcas.get("carrinho") or [], "fundo": marcas.get("fundo") or [], "raio": raio}


def _gc_guiado(rgb, roi, marcas, iteracoes=8, tolerancia_anel=6.0):
    """
    GrabCut guiado pelo usuário.

    - Fora da caixa: fundo certo. Dentro: "provavelmente carrinho".
    - Prior do anel: dentro da caixa, pixels de cor quase idêntica à de algum
      pixel do anel logo fora da caixa são marcados "provavelmente fundo"
      (o GrabCut ainda pode corrigir). Ajuda quando o fundo é variado.
    - Traços do usuário valem como verdade (prevalecem sobre tudo).
    Devolve (mask bool, etapas_debug).
    """
    altura, largura = rgb.shape[:2]
    marcas = _validar_marcas(marcas)
    if roi is None:
        x0, y0, x1, y1 = 0, 0, largura, altura
    else:
        x0, y0, x1, y1 = _caixa_em_pixels(roi, altura, largura)

    gc = np.full((altura, largura), cv2.GC_BGD, dtype=np.uint8)
    gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
    caixa = np.zeros((altura, largura), bool)
    caixa[y0:y1, x0:x1] = True

    # prior do anel externo (só existe se a caixa não ocupa a imagem toda)
    k = max(6, int(0.06 * max(x1 - x0, y1 - y0)))
    anel = (cv2.dilate(caixa.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0) & ~caixa
    if anel.sum() >= 200:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        amostra = lab[anel]
        amostra = amostra[:: max(1, len(amostra) // 4000)]
        dist, _ = _spatial.cKDTree(amostra).query(lab[y0:y1, x0:x1].reshape(-1, 3))
        gc[y0:y1, x0:x1][dist.reshape(y1 - y0, x1 - x0) < tolerancia_anel] = cv2.GC_PR_BGD

    fg = bg = np.zeros((altura, largura), bool)
    if marcas:
        fg = _rasterizar_marcas(marcas["carrinho"], altura, largura, marcas["raio"])
        bg = _rasterizar_marcas(marcas["fundo"], altura, largura, marcas["raio"])
        gc[bg] = cv2.GC_BGD
        gc[fg] = cv2.GC_FGD          # se os dois se cruzam, "carrinho" vence

    if not ((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)).any():   # caixa = imagem toda
        gc[:2, :] = gc[-2:, :] = gc[:, :2] = gc[:, -2:] = cv2.GC_BGD
    if not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
        raise ValueError("A caixa (roi) ficou sem nenhuma região de carrinho.")

    cv2.setRNGSeed(12345)  # resultado reprodutível
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iteracoes, cv2.GC_INIT_WITH_MASK)
    mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    marcas_viz = np.where(fg, 255, np.where(bg, 128, 0)).astype(np.uint8)
    return mask, {"1_roi": caixa, "2_marcas": marcas_viz, "3_grabcut": mask}


def _mascara_guiada(rgb, roi, marcas, manter_maior_objeto, area_minima_relativa):
    """GrabCut guiado + limpeza. É o que a janela de seleção mostra como prévia."""
    mask, etapas = _gc_guiado(rgb, roi, marcas)
    mask = _limpar_mascara(mask, manter_maior_objeto, area_minima_relativa)
    etapas["4_limpa"] = mask
    return mask, etapas


# -------------------------------------------------------------------
# Modo centrado (fotos reais: o carrinho está no meio, o fundo é variado)
# -------------------------------------------------------------------

def _mascara_centrada(rgb, area_minima_relativa, margem=0.03, borda=0.03, iteracoes=6):
    """
    GrabCut com a hipótese "o carrinho está no centro da foto".

    - Faixa de `borda` ao redor da imagem: fundo certo.
    - Miolo (menos `margem` em cada lado): "provavelmente carrinho".
    O GrabCut aprende as cores do fundo na periferia e remove do miolo o que
    for parecido (mesa, parede). Não depende de o fundo ser uniforme.
    Limitação: o que estiver na faixa da borda é cortado.
    """
    altura, largura = rgb.shape[:2]
    gc = np.full((altura, largura), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[int(altura * margem):int(np.ceil(altura * (1 - margem))),
       int(largura * margem):int(np.ceil(largura * (1 - margem)))] = cv2.GC_PR_FGD
    b = max(2, int(borda * min(altura, largura)))
    gc[:b, :] = gc[-b:, :] = gc[:, :b] = gc[:, -b:] = cv2.GC_BGD

    cv2.setRNGSeed(12345)  # resultado reprodutível
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iteracoes, cv2.GC_INIT_WITH_MASK)
    mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return mask


def _remover_espinhos(mask):
    """
    Remove linhas finas coladas na máscara (ex.: contorno desfocado de um
    objeto do fundo que o GrabCut pegou). Abre com um kernel de ~0,3% da
    imagem e devolve só o que está colado ao miolo grosso. Peças realmente
    finas (< ~3 px em 1024) somem; uma asa de 4 px é mantida. Só é usado no
    modo centrado (não estica a caixa do carrinho com contornos de fundo).
    """
    k = int(round(0.003 * max(mask.shape))) | 1
    k = max(3, k)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    aberta = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, nucleo)
    if not aberta.any():
        return mask                      # tudo era fino: não mexe
    return mask & (cv2.dilate(aberta, nucleo) > 0)


def _penalidade(mask):
    """
    Quão SUSPEITA é uma máscara de carrinho (0 = plausível; maior = pior).
    Soma: partes grandes além da primeira + bordas da imagem tocadas além da
    primeira + 2 se a caixa da máscara cobre quase a imagem toda.
    """
    n, _, est, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    grandes = int((est[1:, cv2.CC_STAT_AREA] / mask.size >= 0.01).sum()) if n > 1 else 0
    bordas = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    ys, xs = np.nonzero(mask)
    caixa = ((xs.max() - xs.min()) * (ys.max() - ys.min()) / mask.size) if len(xs) else 0.0
    return max(0, grandes - 1) + max(0, bordas - 1) + (2 if caixa > 0.85 else 0)


def _variedade_das_bordas(rgb, mask, limite=20.0):
    """
    Mede quanta coisa há nas bordas da imagem. Devolve (nao_explicado, total):
      total          = fração das bordas cuja cor foge de um fundo suave
                       (superfície de grau 2 ajustada de forma robusta);
      nao_explicado  = parte disso que NÃO está coberta pela máscara.
    Fundo simples (mesmo com gradiente): ambos ~0. Carrinho encostado na
    borda: total alto, nao_explicado ~0. Gente/roupas/objetos no fundo: total
    alto e, se a máscara errou, nao_explicado alto.
    Limite: se o carrinho ocupa mais de ~30% da borda, o ajuste do fundo fica
    pouco confiável (mas aí a detecção por borda já falha antes).
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    altura, largura = lab.shape[:2]
    m = max(5, int(min(altura, largura) * 0.03))
    m = max(1, min(m, altura // 2, largura // 2))
    anel = np.zeros((altura, largura), bool)
    anel[:m] = anel[-m:] = True
    anel[:, :m] = anel[:, -m:] = True

    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    x = x / max(1, largura - 1) * 2 - 1
    y = y / max(1, altura - 1) * 2 - 1
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)[anel]
    B = lab[anel]
    sel = np.ones(len(B), bool)
    for _ in range(3):                         # ajuste robusto: ignora quem destoa
        coef, *_ = np.linalg.lstsq(A[sel], B[sel], rcond=None)
        r = np.linalg.norm(A @ coef - B, axis=1)
        sel = r <= max(8.0, 2.5 * np.median(r[sel]) + 4)
    alto = r > limite
    coberto = (cv2.dilate(mask.astype(np.uint8), np.ones((2 * m + 1, 2 * m + 1), np.uint8)) > 0)[anel]
    return float((alto & ~coberto).mean()), float(alto.mean())


# -------------------------------------------------------------------
# Modo IA (opcional): rede neural local para achar o carrinho
# -------------------------------------------------------------------
#
# Usa o pacote `rembg` (remoção de fundo, modelo U2-Net) rodando NO SEU
# COMPUTADOR; a foto não é enviada para a internet. Instalação:
#     pip install rembg onnxruntime
# Na primeira vez ele baixa o modelo (~170 MB). Sem a caixa, a IA olha a
# imagem inteira; com a caixa (roi), olha só o recorte, o que costuma ser
# bem melhor. Pincéis refinam o resultado da IA com GrabCut.

_SESSOES_IA = {}


def _sessao_ia(modelo="u2net"):
    try:
        from rembg import new_session  # type: ignore
    except ImportError as erro:
        raise ValueError(
            "O modo 'ia' precisa do pacote rembg. Instale com:  pip install rembg onnxruntime  "
            "(na primeira vez ele baixa um modelo de ~170 MB).") from erro
    if modelo not in _SESSOES_IA:
        _SESSOES_IA[modelo] = new_session(modelo)
    return _SESSOES_IA[modelo]


def _refinar_ia_com_marcas(rgb, mask, marcas):
    """Usa a máscara da IA como ponto de partida do GrabCut e aplica os pincéis do usuário."""
    altura, largura = mask.shape
    fg = _rasterizar_marcas(marcas["carrinho"], altura, largura, marcas["raio"])
    bg = _rasterizar_marcas(marcas["fundo"], altura, largura, marcas["raio"])
    gc = np.where(mask, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[bg] = cv2.GC_BGD
    gc[fg] = cv2.GC_FGD
    tem_fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any()
    tem_bg = ((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)).any()
    if not (tem_fg and tem_bg):
        return mask
    cv2.setRNGSeed(12345)  # resultado reprodutível
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                5, cv2.GC_INIT_WITH_MASK)
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def _mascara_ia(rgb, roi, marcas, manter_maior_objeto, area_minima_relativa):
    """Máscara do carrinho feita pela IA (rembg), opcionalmente só dentro da caixa."""
    sessao = _sessao_ia()
    from rembg import remove  # type: ignore
    altura, largura = rgb.shape[:2]
    if roi is None:
        x0, y0, x1, y1 = 0, 0, largura, altura
    else:
        x0, y0, x1, y1 = _caixa_em_pixels(roi, altura, largura)
    try:
        resultado = remove(Image.fromarray(rgb[y0:y1, x0:x1]), session=sessao, only_mask=True)
    except Exception as erro:
        raise ValueError(f"A IA (rembg) falhou: {erro}") from erro

    prob = np.asarray(resultado)
    if prob.ndim == 3:                       # RGBA: usa o canal alfa
        prob = prob[..., -1]
    if prob.shape != (y1 - y0, x1 - x0):
        prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((altura, largura), bool)
    mask[y0:y1, x0:x1] = prob >= 128

    marcas = _validar_marcas(marcas)
    if marcas and mask.any():
        mask = _refinar_ia_com_marcas(rgb, mask, marcas)
    if not mask.any():
        raise ValueError("A IA não encontrou nenhum objeto na imagem (ou dentro da caixa).")
    mask = _limpar_mascara(mask, manter_maior_objeto, area_minima_relativa)
    return mask, {"3_ia": mask, "4_limpa": mask}


def _segmentar_direto(rgb, alpha, usar_grabcut, sensibilidade, manter_maior_objeto,
                      area_minima_relativa, roi=None, marcas=None, modo="auto"):
    """
    Segmenta na resolução recebida. Devolve (mask, etapas_para_debug).

    modo (só na detecção automática):
      "borda"  : estima o fundo pelas bordas (ótimo em fundo simples/uniforme)
      "centro" : GrabCut assumindo o carrinho no centro (fotos com fundo variado)
      "auto"   : usa "borda"; se o resultado parecer errado (partes soltas,
                 toca várias bordas), tenta "centro" e fica com o mais plausível
      "ia"     : rede neural local (rembg); com `roi`, só dentro da caixa
    """
    if modo not in ("auto", "borda", "centro", "ia"):
        raise ValueError("modo deve ser 'auto', 'borda', 'centro' ou 'ia'.")
    if alpha is None and modo == "ia":
        return _mascara_ia(rgb, roi, marcas, manter_maior_objeto, area_minima_relativa)
    if alpha is None and (roi is not None or marcas):
        return _mascara_guiada(rgb, roi, marcas, manter_maior_objeto, area_minima_relativa)

    if alpha is None and modo == "centro":
        mask = _limpar_mascara(_remover_espinhos(_mascara_centrada(rgb, area_minima_relativa)),
                               manter_maior_objeto, area_minima_relativa)
        if not mask.any():
            raise ValueError("Nenhum objeto foi encontrado na imagem.")
        return mask, {"3_grabcut": mask, "4_limpa": mask}

    centro = lambda: _limpar_mascara(_remover_espinhos(_mascara_centrada(rgb, area_minima_relativa)),
                                     manter_maior_objeto, area_minima_relativa)
    if alpha is not None:
        forte = fraca = alpha >= 128           # PNG com fundo transparente
    else:
        try:
            forte, fraca = _mascara_por_cor(rgb, sensibilidade)
        except ValueError:
            # Sem contraste com o fundo estimado pelas bordas (ex.: o carrinho
            # ocupa boa parte da borda). No automático, tenta o modo centrado.
            if modo != "auto" or not usar_grabcut:
                raise
            mask = centro()
            if not mask.any():
                raise
            warnings.warn("Não foi possível estimar o fundo pelas bordas; usei o modo "
                          "centrado (assume o carrinho no meio da foto). Confira o resultado.")
            return mask, {"3_grabcut": mask, "4_limpa": mask}
    etapas = {"1_forte": forte, "2_fraca": fraca}

    if not fraca.any():
        raise ValueError("Nenhum objeto foi encontrado na imagem.")

    mask = fraca
    if usar_grabcut and alpha is None:
        mask = _refinar_com_grabcut(rgb, forte, fraca)
    etapas["3_grabcut"] = mask

    if alpha is None:
        # quinas borradas entre duas regiões de fundo (parede/chão, mesa/borda)
        # viram linhas de 1-2 px "parecidas com carrinho": removê-las
        mask = _remover_espinhos(mask)
    mask = _limpar_mascara(mask, manter_maior_objeto, area_minima_relativa)
    etapas["4_limpa"] = mask

    # A detecção pela borda é confiável? Desconfia se a máscara tem partes
    # soltas / toca várias bordas, OU se as bordas da imagem têm muita coisa
    # que a máscara não explica (gente, roupa, objetos no fundo).
    if alpha is None and modo == "auto" and usar_grabcut:
        pen_borda = _penalidade(mask)
        nao_explicado, total = _variedade_das_bordas(rgb, mask)
        heterogeneo = nao_explicado > 0.2           # bordas com muita coisa que a máscara não cobre
        # Partes soltas só são suspeitas se o fundo também for variado (um
        # carrinho com rodas separadas num fundo liso é perfeitamente normal).
        if heterogeneo or (pen_borda > 0 and total > 0.2):
            centrada = centro()
            if heterogeneo and mask.mean() < 0.01 and centrada.mean() < 0.01:
                # bordas cheias de coisa e as DUAS tentativas acharam quase nada:
                # melhor um erro claro do que uma máscara lixo silenciosa
                raise ValueError(
                    "Não foi possível distinguir o carrinho do fundo: ele parece ocupar boa "
                    "parte das bordas, ou o fundo é muito variado. Deixe um pouco de fundo ao "
                    "redor dele ou aponte o carrinho (seleção guiada, roi=...).")
            if centrada.any():
                concordam = (mask & centrada).sum() / max(1, (mask | centrada).sum()) >= 0.9
                pen_c = _penalidade(centrada)
                melhor = pen_c < pen_borda or (heterogeneo and pen_c <= pen_borda)
                if melhor and not concordam:
                    motivo = ("o fundo tem muita coisa nas bordas" if heterogeneo
                              else "havia partes soltas ou muitas bordas tocadas")
                    warnings.warn(
                        f"A detecção pela borda parecia errada ({motivo}); usei o modo "
                        "centrado, que assume o carrinho no meio da foto. Confira o "
                        "resultado; se precisar, aponte o carrinho (seleção guiada).")
                    etapas["5_alternativa_borda"] = mask
                    return centrada, {**etapas, "3_grabcut": centrada, "4_limpa": centrada}
    return mask, etapas


def _ampliar(m, altura, largura):
    """Amplia uma máscara/etapa (bool ou uint8) de volta ao tamanho original."""
    if m.shape == (altura, largura):
        return m
    if m.dtype == bool:
        return cv2.resize(m.astype(np.float32), (largura, altura),
                          interpolation=cv2.INTER_LINEAR) >= 0.5
    return cv2.resize(m, (largura, altura), interpolation=cv2.INTER_NEAREST)


def _segmentar(rgb, alpha, usar_grabcut, sensibilidade, manter_maior_objeto,
               area_minima_relativa, roi=None, marcas=None, modo="auto",
               lado_segmentacao=512):
    """
    Segmenta o carrinho. Devolve (mask, etapas_para_debug) no tamanho de `rgb`.

    Por velocidade, a segmentação roda numa cópia reduzida (lado maior =
    `lado_segmentacao`) e a máscara é ampliada de volta. O GrabCut fica
    MUITO mais lento com a resolução (ex.: 15 s em 768x1024 contra ~2 s em
    384x512). Peças finas (hastes, asas de poucos pixels) podem se perder; suba
    `lado_segmentacao` (ex.: 1024) se precisar delas.
    """
    altura, largura = rgb.shape[:2]
    escala = lado_segmentacao / max(altura, largura)
    if alpha is not None or escala >= 1:          # silhueta transparente é barata
        return _segmentar_direto(rgb, alpha, usar_grabcut, sensibilidade,
                                 manter_maior_objeto, area_minima_relativa,
                                 roi=roi, marcas=marcas, modo=modo)
    pequeno = cv2.resize(rgb, (max(8, round(largura * escala)), max(8, round(altura * escala))),
                         interpolation=cv2.INTER_AREA)
    mask, etapas = _segmentar_direto(pequeno, None, usar_grabcut, sensibilidade,
                                     manter_maior_objeto, area_minima_relativa,
                                     roi=roi, marcas=marcas, modo=modo)
    return (_ampliar(mask, altura, largura),
            {nome: _ampliar(e, altura, largura) for nome, e in etapas.items()})


# -------------------------------------------------------------------
# Conversão para o domínio computacional
# -------------------------------------------------------------------

def _reduzir_preservando_pecas_finas(mask, nova_largura, nova_altura):
    """
    Redimensiona uma máscara bool para a malha.

    Partes grossas: limiar 0,5 (não infla a área). Partes finas (mais
    estreitas que ~1,5 célula) usam limiar 0,2, para não se fragmentarem
    nem sumirem. Medido: área entre 1,00x e 1,02x da esperada em malhas
    de >= 100 células de largura, sem fragmentar a asa fina.
    """
    h, w = mask.shape
    area = lambda m: cv2.resize(m.astype(np.float32), (nova_largura, nova_altura),
                                interpolation=cv2.INTER_AREA)
    grosso = area(mask) >= 0.5

    escala = min(nova_largura / w, nova_altura / h)
    if escala < 1:
        k = int(round(1.5 / escala)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        aberta = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        grosso |= area(mask & ~aberta) >= 0.2
    return ndimage.binary_fill_holes(grosso, structure=_8_VIZINHOS)


def _numero_de_componentes(mask):
    return ndimage.label(mask, structure=_8_VIZINHOS)[1]


def _numero(valor, nome, minimo, maximo, minimo_exclusivo=False):
    """Valida um número real finito em um intervalo."""
    if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
        raise ValueError(f"{nome} deve ser um número (recebido: {valor!r}).")
    valor = float(valor)
    dentro = (valor > minimo if minimo_exclusivo else valor >= minimo) and valor <= maximo
    if not np.isfinite(valor) or not dentro:
        abre = "(" if minimo_exclusivo else "["
        raise ValueError(f"{nome} deve estar em {abre}{minimo}, {maximo}] (recebido: {valor!r}).")
    return valor


def _inteiro_positivo(valor, nome):
    """Aceita só inteiros positivos de verdade (rejeita 100.5, True, '100')."""
    if isinstance(valor, bool) or not isinstance(valor, numbers.Integral) or valor < 1:
        raise ValueError(f"{nome} deve ser um inteiro positivo (recebido: {valor!r}).")
    return int(valor)


def _salvar_debug(debug_dir, rgb, etapas, mask_img, mask_final):
    os.makedirs(debug_dir, exist_ok=True)
    salvar = lambda nome, a: _salvar_png(os.path.join(debug_dir, nome), a)
    salvar("0_entrada.png", rgb)
    for nome, m in etapas.items():
        salvar(f"{nome}.png", m)
    # sobreposição: carrinho detectado em vermelho sobre a imagem original
    sobre = rgb.astype(np.float32)
    sobre[mask_img] = 0.5 * sobre[mask_img] + 0.5 * np.array([255, 0, 0], np.float32)
    salvar("5_sobreposicao.png", sobre.astype(np.uint8))
    salvar("6_final_na_malha.png", mask_final)


# -------------------------------------------------------------------
# Função principal
# -------------------------------------------------------------------

def criar_mascara(
    caminho_imagem,
    largura,
    altura,
    fracao_dominio=0.4,
    posicao_x=0.35,
    posicao_y=0.5,
    usar_grabcut=True,
    sensibilidade=0.5,
    manter_maior_objeto=False,
    area_minima_relativa=0.005,
    lado_maximo=1024,
    salvar_em=None,
    debug_dir=None,
    espelhar_horizontal=False,
    ocupacao_maxima=0.85,
    roi=None,
    marcas=None,
    modo="auto",
    lado_segmentacao=512,
):
    """
    Converte uma imagem/silhueta em uma máscara booleana para o solver LBM.

    Parâmetros
    ----------
    caminho_imagem : str | os.PathLike
    largura, altura : int
        Tamanho da malha (nº de células em x e em y).
    fracao_dominio : float em (0, 1]
        Fração do domínio que o carrinho pode ocupar (na largura ou na
        altura, o que limitar primeiro). Deixa espaço para o escoamento.
    posicao_x, posicao_y : float em [0, 1]
        Posição do CENTRO do carrinho no domínio (0 = esquerda/topo).
    usar_grabcut : bool
        Refina a segmentação com GrabCut (ajuda em fundos com textura).
    sensibilidade : float em (0, 1]
        Maior = mais permissivo (pega partes claras, mas também mais sombra).
        LIMITAÇÃO CONHECIDA: sombras no chão (só mais escuras que o fundo)
        são tratadas como carrinho. Use fotos sem sombra marcada ou silhuetas.
    manter_maior_objeto : bool
        True mantém só o maior pedaço. Padrão False: preserva rodas e asas
        desconectadas, descartando só componentes minúsculos.
    area_minima_relativa : float em [0, 1)
        Com manter_maior_objeto=False, componentes menores que esta fração
        do maior são tratados como sujeira.
    lado_maximo : int
        Imagens maiores (em pixels) são reduzidas antes de processar.
    salvar_em : str | None
        Salva a máscara final (na malha) como PNG.
    debug_dir : str | None
        Salva entrada, etapas, sobreposição e resultado final nessa pasta.
    espelhar_horizontal : bool
        Espelha o carrinho (frente para a esquerda <-> direita).
    ocupacao_maxima : float | None
        Se a máscara cobrir mais que esta fração da imagem, assume que o
        fundo foi segmentado e levanta ValueError. None desativa.
        (Só vale na detecção automática.)
    roi : (x0, y0, x1, y1) | None
        Caixa em volta do carrinho, em frações (0 a 1) da imagem. Use quando
        o fundo tem muita coisa: a detecção deixa de ser automática e passa
        a ser guiada. Veja selecionar_carrinho().
    lado_segmentacao : int
        A segmentação roda numa cópia reduzida da imagem (lado maior com esse
        tamanho) para ser rápida; a máscara é ampliada de volta. Aumente (ex.:
        1024) se precisar de peças bem finas (hastes, asas); fica mais lento.
    modo : "auto" | "borda" | "centro" | "ia"
        "ia" usa uma rede neural LOCAL (pacote rembg: pip install rembg
        onnxruntime); com roi, só olha dentro da caixa. Como detectar o carrinho sem seleção do usuário. "borda": estima o
        fundo pelas bordas (ideal para fundo simples). "centro": GrabCut
        supondo o carrinho no meio da foto (fotos com fundo variado, pessoas,
        mesa...). "auto" (padrão): começa por "borda" e, se o resultado parecer
        errado, compara com "centro" e usa o mais plausível.
    marcas : dict | None
        Pincéis do usuário: {"carrinho": [traco...], "fundo": [traco...],
        "raio": 0.008}; traco = lista de pontos (x, y) em frações. Pode ser
        usado com ou sem roi.

    Retorna
    -------
    mask : np.ndarray de bool, shape (altura, largura)
        True = sólido (carrinho), False = fluido (ar).
    """
    # --- validação ---------------------------------------------------
    largura = _inteiro_positivo(largura, "largura")
    altura = _inteiro_positivo(altura, "altura")
    lado_maximo = _inteiro_positivo(lado_maximo, "lado_maximo")
    lado_segmentacao = _inteiro_positivo(lado_segmentacao, "lado_segmentacao")
    fracao_dominio = _numero(fracao_dominio, "fracao_dominio", 0, 1, True)
    sensibilidade = _numero(sensibilidade, "sensibilidade", 0, 1, True)
    posicao_x = _numero(posicao_x, "posicao_x", 0, 1)
    posicao_y = _numero(posicao_y, "posicao_y", 0, 1)
    area_minima_relativa = _numero(area_minima_relativa, "area_minima_relativa", 0, 0.999999)
    if ocupacao_maxima is not None:
        ocupacao_maxima = _numero(ocupacao_maxima, "ocupacao_maxima", 0, 1, True)

    if not os.path.isfile(caminho_imagem):
        raise FileNotFoundError(f"Imagem não encontrada (ou não é um arquivo): {caminho_imagem!r}")

    # --- 1. carregar e segmentar --------------------------------------
    try:
        rgb, alpha = _carregar_imagem(caminho_imagem, lado_maximo)
    except (OSError, ValueError, Image.DecompressionBombError) as erro:
        dica = ""
        if isinstance(erro, UnidentifiedImageError):
            dica = (" Formato não reconhecido. Para HEIC/HEIF instale 'pillow-heif'; "
                    "SVG e PDF não são suportados (converta para PNG ou JPG).")
        raise ValueError(
            f"Não foi possível abrir {caminho_imagem!r} como imagem: {erro}.{dica}") from erro

    guiada = alpha is None and (roi is not None or bool(marcas))
    if alpha is not None and (roi is not None or marcas):
        warnings.warn("A imagem tem fundo transparente; roi/marcas foram ignorados.")

    mask_img, etapas = _segmentar(rgb, alpha, usar_grabcut, sensibilidade,
                                  manter_maior_objeto, area_minima_relativa,
                                  roi=roi, marcas=marcas, modo=modo,
                                  lado_segmentacao=lado_segmentacao)

    # --- 2. verificações de plausibilidade ----------------------------
    if mask_img.sum() < 25:
        raise ValueError("Nenhum objeto foi encontrado na imagem.")
    if alpha is None and not guiada and modo != "ia":
        ocupacao = mask_img.mean()
        if ocupacao_maxima is not None and ocupacao > ocupacao_maxima:
            raise ValueError(
                f"A máscara cobre {ocupacao:.0%} da imagem: provavelmente o fundo "
                "foi segmentado como carrinho. Use debug_dir para ver as etapas "
                "(ou ocupacao_maxima=None se isso for realmente esperado).")
        if ocupacao < 0.002:
            warnings.warn(f"A máscara cobre só {ocupacao:.2%} da imagem; o carrinho "
                          "pode ter sido segmentado incompletamente.")
        ys_, xs_ = np.nonzero(mask_img)
        caixa_cobre = ((xs_.max() - xs_.min()) * (ys_.max() - ys_.min())) / mask_img.size
        bordas = (mask_img[0].any(), mask_img[-1].any(), mask_img[:, 0].any(), mask_img[:, -1].any())
        if sum(bordas) >= 3 or caixa_cobre > 0.85:
            warnings.warn(
                "A máscara toca várias bordas da imagem: o fundo provavelmente tem muitos "
                "objetos e foi confundido com o carrinho. Use a seleção guiada "
                "(roi=..., ou 'python geometry.py --selecionar') para apontar o carrinho.")

    # --- 3. recorta e redimensiona sem distorcer ----------------------
    ys, xs = np.nonzero(mask_img)
    mask = mask_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if espelhar_horizontal:
        mask = mask[:, ::-1]
    h, w = mask.shape

    escala = min((largura * fracao_dominio) / w, (altura * fracao_dominio) / h)
    nova_largura = int(np.clip(round(w * escala), 1, largura))
    nova_altura = int(np.clip(round(h * escala), 1, altura))
    reduzida = _reduzir_preservando_pecas_finas(mask, nova_largura, nova_altura)

    # --- 4. posiciona na malha ----------------------------------------
    mask_final = np.zeros((altura, largura), dtype=bool)
    x0 = int(np.clip(round(posicao_x * largura - nova_largura / 2), 0, largura - nova_largura))
    y0 = int(np.clip(round(posicao_y * altura - nova_altura / 2), 0, altura - nova_altura))
    mask_final[y0:y0 + nova_altura, x0:x0 + nova_largura] = reduzida

    # --- 5. avisos sobre a qualidade da geometria na malha ------------
    esperada = mask.sum() * (nova_largura / w) * (nova_altura / h)
    razao = reduzida.sum() / max(esperada, 1.0)
    if not 0.8 <= razao <= 1.25:
        warnings.warn(
            f"A área do carrinho na malha difere {razao - 1:+.0%} da esperada: a "
            "resolução é baixa para esta geometria. Considere uma malha maior.")
    antes, depois = _numero_de_componentes(mask), _numero_de_componentes(reduzida)
    if antes != depois:
        warnings.warn(
            f"O número de partes do carrinho mudou na redução (imagem: {antes}, "
            f"malha: {depois}): peças finas se fragmentaram, sumiram ou se fundiram. "
            "A resolução é baixa para esta geometria.")
    if mask_final[0, :].any() or mask_final[-1, :].any() \
            or mask_final[:, 0].any() or mask_final[:, -1].any():
        warnings.warn("O carrinho encosta na borda do domínio; isso conflita com "
                      "entrada/saída/paredes do solver. Reduza fracao_dominio.")

    # --- 6. saídas opcionais -------------------------------------------
    if debug_dir is not None:
        _salvar_debug(debug_dir, rgb, etapas, mask_img, mask_final)
    if salvar_em is not None:
        _salvar_png(salvar_em, mask_final)

    return mask_final


# -------------------------------------------------------------------
# Escolha da imagem pelo explorador de arquivos
# -------------------------------------------------------------------

_EXTENSOES_DIALOGO = (".png", ".jpg", ".jpeg", ".jfif", ".jpe", ".bmp", ".gif", ".tif",
                      ".tiff", ".webp", ".avif", ".heic", ".heif", ".ico", ".ppm", ".pgm",
                      ".tga")


def _trazer_para_frente(janela):
    """
    Faz a janela aparecer NA FRENTE e com foco, centralizada na tela.
    (No Windows, janelas novas costumam abrir escondidas atrás das outras e
    só aparecem ao clicar na barra de tarefas.)
    """
    try:
        janela.update_idletasks()
        largura, altura = janela.winfo_width(), janela.winfo_height()
        x = max(0, (janela.winfo_screenwidth() - largura) // 2)
        y = max(0, (janela.winfo_screenheight() - altura) // 3)
        janela.geometry(f"+{x}+{y}")
        janela.deiconify()
        janela.attributes("-topmost", True)      # força ficar na frente...
        janela.lift()
        janela.focus_force()
        janela.after(500, lambda: janela.attributes("-topmost", False))   # ...e depois libera
    except Exception:
        pass


def escolher_imagem():
    """
    Abre o explorador de arquivos para o usuário escolher a imagem.

    Devolve o caminho escolhido, ou None se a janela for cancelada.
    Se não houver tkinter ou interface gráfica (ex.: servidor, SSH),
    pede para digitar o caminho no terminal.
    """
    _pre_carregar_bibliotecas()    # enquanto você escolhe, OpenCV/SciPy carregam ao fundo
    try:
        import tkinter as tk
        from tkinter import filedialog
        raiz = tk.Tk()
    except Exception:
        caminho = input("Digite o caminho da imagem: ").strip().strip('"')
        return caminho or None

    raiz.withdraw()
    try:
        raiz.attributes("-topmost", True)   # janela na frente das demais
    except tk.TclError:
        pass
    try:
        caminho = filedialog.askopenfilename(
            title="Escolha a imagem do carrinho",
            filetypes=[
                ("Imagens", " ".join("*" + e for e in _EXTENSOES_DIALOGO)),
                ("Todos os arquivos", "*.*"),
            ],
        )
    finally:
        raiz.destroy()
    return caminho or None


# -------------------------------------------------------------------
# Janela de seleção do carrinho (para fundos com muita coisa)
# -------------------------------------------------------------------

class _EstadoSelecao:
    """
    Estado da seleção do usuário (caixa e traços de pincel), em FRAÇÕES da
    imagem. Não depende de janela nenhuma, por isso é testável.
    """

    def __init__(self, raio=0.008):
        self.roi = None
        self.raio = raio
        self.tracos = {"carrinho": [], "fundo": []}
        self._atual = None

    def definir_caixa(self, x0, y0, x1, y1):
        """Define a caixa. Devolve False (e ignora) se for pequena demais."""
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            return False
        self.roi = (x0, y0, x1, y1)
        return True

    def novo_traco(self, tipo, x, y):
        self._atual = [(x, y)]
        self.tracos[tipo].append(self._atual)

    def estender_traco(self, x, y):
        if self._atual is not None:
            self._atual.append((x, y))

    def limpar_marcas(self):
        self.tracos = {"carrinho": [], "fundo": []}
        self._atual = None

    def tem_marcas(self):
        return bool(self.tracos["carrinho"] or self.tracos["fundo"])

    def marcas(self):
        return {"carrinho": [list(t) for t in self.tracos["carrinho"]],
                "fundo": [list(t) for t in self.tracos["fundo"]],
                "raio": self.raio}

    def para_dict(self):
        """Formato aceito por criar_mascara(roi=..., marcas=...) e por JSON."""
        return {"roi": list(self.roi) if self.roi else None,
                "marcas": self.marcas() if self.tem_marcas() else None}


def carregar_selecao(caminho_json):
    """Lê uma seleção salva (JSON) -> dict com 'roi' e 'marcas'."""
    try:
        with open(caminho_json, encoding="utf-8") as f:
            dados = json.load(f)
    except (OSError, json.JSONDecodeError) as erro:
        raise ValueError(f"Não foi possível ler a seleção {caminho_json!r}: {erro}") from erro
    if not isinstance(dados, dict):
        raise ValueError("O arquivo de seleção deve conter um objeto JSON com 'roi' e/ou 'marcas'.")
    return {"roi": dados.get("roi"), "marcas": dados.get("marcas")}


def salvar_selecao(selecao, caminho_json):
    """Salva a seleção em JSON para reutilizar (mesma geometria, sempre)."""
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(selecao, f, indent=2)


def selecionar_carrinho(caminho_imagem, lado_maximo=1024, usar_grabcut=True,
                        sensibilidade=0.5, manter_maior_objeto=False,
                        area_minima_relativa=0.005, modo="auto", lado_segmentacao=512):
    """
    Abre uma janela para você apontar onde está o carrinho.

    Como usar a janela
      - A máscara automática aparece em vermelho assim que abre. Se estiver
        certa, clique em "Concluir".
      - Se errou (fundo bagunçado): modo "Caixa" e ARRASTE um retângulo em
        volta do carrinho inteiro; clique em "Calcular máscara".
      - Ainda errou? Use o pincel verde (isto é carrinho) e o vermelho (isto é
        fundo) sobre os erros e calcule de novo. Poucos traços bastam.

    Devolve:
      dict {"roi": ..., "marcas": ...}  -> passe a criar_mascara(**dict)
      {}                                -> use a detecção automática
      None                              -> o usuário cancelou
    Levanta RuntimeError se não houver tkinter / interface gráfica.
    """
    try:
        import tkinter as tk
        from PIL import ImageTk
    except Exception as erro:
        raise RuntimeError(f"tkinter indisponível: {erro}") from erro

    rgb, alpha = _carregar_imagem(caminho_imagem, lado_maximo)
    if alpha is not None:
        return {}                      # silhueta transparente: nada a apontar

    h, w = rgb.shape[:2]
    escala = min(1.0, 1000 / w, 620 / h)
    dw, dh = max(2, int(w * escala)), max(2, int(h * escala))
    base = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)

    try:
        raiz = tk.Tk()
    except Exception as erro:
        raise RuntimeError(f"não foi possível abrir a janela: {erro}") from erro
    raiz.title("Selecionar o carrinho")

    estado = _EstadoSelecao()
    resultado = {"valor": None}
    ini = [None]       # início do arraste da caixa
    ultimo = [None]    # último ponto do pincel
    cores = {"carrinho": "#00c800", "fundo": "#e00000"}

    ferramenta = tk.StringVar(value="caixa")
    status = tk.StringVar(value="Abrindo...")
    espessura = tk.IntVar(value=max(2, int(round(2 * estado.raio * max(dw, dh)))))
    usar_ia = tk.BooleanVar(value=(modo == "ia"))

    def px_pincel():
        return max(1, int(espessura.get()))

    def mostrar(mask=None):
        quadro = np.array(base)
        if mask is not None:
            m = cv2.resize(mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            quadro = quadro.astype(np.float32)
            quadro[m] = 0.55 * quadro[m] + 0.45 * np.array([255, 40, 40], np.float32)
            quadro = np.ascontiguousarray(quadro.astype(np.uint8))
            contornos, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(quadro, contornos, -1, (255, 255, 0), 1)
        foto = ImageTk.PhotoImage(Image.fromarray(quadro))
        canvas.itemconfig(item_imagem, image=foto)
        canvas.imagem_atual = foto           # evita que o Python apague a imagem
        canvas.tag_raise("caixa")
        canvas.tag_raise("traco")

    def calcular():
        status.set("Calculando...")
        raiz.update_idletasks()
        try:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                guiado = estado.roi is not None or estado.tem_marcas()
                # mesmo caminho do resultado final: o que você aprova é o que sai
                mask, _ = _segmentar(rgb, None, usar_grabcut, sensibilidade,
                                     manter_maior_objeto, area_minima_relativa,
                                     roi=estado.roi, marcas=estado.marcas() if guiado else None,
                                     modo="ia" if usar_ia.get() else modo,
                                     lado_segmentacao=lado_segmentacao)
                dica = ("Pronto. Corrija com os pincéis e calcule de novo, ou clique em Concluir."
                        if guiado else
                        "Máscara automática. Se errou, desenhe uma caixa em volta do carrinho.")
            nossos = [a for a in avisos if issubclass(a.category, UserWarning)]
            if nossos:                       # ex.: "usei o modo centrado..."
                dica += " Aviso: " + str(nossos[0].message)
        except ValueError as erro:
            status.set(f"Não foi possível: {erro}")
            return
        try:
            mostrar(mask)
        except Exception as erro:      # erro de desenho: mostra na janela, não esconde
            status.set(f"Erro ao desenhar a prévia: {erro}")
            return
        status.set(dica)

    def ponto(ev):
        return min(max(ev.x, 0), dw - 1), min(max(ev.y, 0), dh - 1)

    def ao_pressionar(ev):
        x, y = ponto(ev)
        if ferramenta.get() == "caixa":
            ini[0] = (x, y)
            canvas.delete("caixa")
            canvas.create_rectangle(x, y, x, y, outline="yellow", width=2, tags="caixa")
        else:
            tipo = ferramenta.get()
            estado.novo_traco(tipo, x / (dw - 1), y / (dh - 1))
            ultimo[0] = (x, y)
            r = px_pincel() / 2
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=cores[tipo],
                               outline="", tags="traco")

    def ao_arrastar(ev):
        x, y = ponto(ev)
        if ferramenta.get() == "caixa":
            if ini[0] is not None:
                canvas.coords("caixa", ini[0][0], ini[0][1], x, y)
        elif ultimo[0] is not None:
            tipo = ferramenta.get()
            estado.estender_traco(x / (dw - 1), y / (dh - 1))
            canvas.create_line(ultimo[0][0], ultimo[0][1], x, y, fill=cores[tipo],
                               width=px_pincel(), capstyle="round", tags="traco")
            ultimo[0] = (x, y)

    def ao_soltar(ev):
        if ferramenta.get() == "caixa" and ini[0] is not None:
            x, y = ponto(ev)
            ok = estado.definir_caixa(min(ini[0][0], x) / dw, min(ini[0][1], y) / dh,
                                      max(ini[0][0], x) / dw, max(ini[0][1], y) / dh)
            ini[0] = None
            if ok:
                status.set("Caixa definida. Clique em 'Calcular máscara'.")
            else:
                canvas.delete("caixa")
                status.set("Caixa pequena demais: arraste em volta do carrinho inteiro.")
        ultimo[0] = None

    def limpar_pinceis():
        estado.limpar_marcas()
        canvas.delete("traco")
        status.set("Pincéis removidos.")

    def concluir():
        resultado["valor"] = estado.para_dict() if (estado.roi or estado.tem_marcas()) else {}
        raiz.destroy()

    def cancelar():
        resultado["valor"] = None
        raiz.destroy()

    def ao_mudar_pincel(valor=None):
        estado.raio = px_pincel() / (2.0 * max(dw, dh))

    barra = tk.Frame(raiz)
    barra.pack(side="top", fill="x")
    for texto, valor in (("Caixa (arraste em volta do carrinho)", "caixa"),
                         ("Pincel verde: é carrinho", "carrinho"),
                         ("Pincel vermelho: é fundo", "fundo")):
        tk.Radiobutton(barra, text=texto, variable=ferramenta, value=valor).pack(side="left")
    tk.Scale(barra, from_=2, to=40, orient="horizontal", label="Tamanho do pincel",
             variable=espessura, command=ao_mudar_pincel).pack(side="left")
    tk.Checkbutton(barra, text="Usar IA (rembg)", variable=usar_ia).pack(side="left", padx=4)
    tk.Button(barra, text="Calcular máscara", command=calcular).pack(side="left", padx=4)
    tk.Button(barra, text="Limpar pincéis", command=limpar_pinceis).pack(side="left")
    tk.Button(barra, text="Concluir", command=concluir).pack(side="left", padx=4)
    tk.Button(barra, text="Cancelar", command=cancelar).pack(side="left")
    tk.Label(raiz, textvariable=status, anchor="w").pack(side="bottom", fill="x")

    canvas = tk.Canvas(raiz, width=dw, height=dh, cursor="crosshair")
    canvas.pack(side="top")
    item_imagem = canvas.create_image(0, 0, anchor="nw")
    canvas.bind("<ButtonPress-1>", ao_pressionar)
    canvas.bind("<B1-Motion>", ao_arrastar)
    canvas.bind("<ButtonRelease-1>", ao_soltar)
    raiz.protocol("WM_DELETE_WINDOW", cancelar)

    mostrar(None)
    _trazer_para_frente(raiz)
    raiz.after(150, calcular)          # mostra a janela e já calcula a automática
    raiz.mainloop()
    return resultado["valor"]


# -------------------------------------------------------------------
# Prévia do resultado (mostrar a máscara na tela)
# -------------------------------------------------------------------

def montar_previa(caminho_imagem, mask, altura=420):
    """
    Monta uma imagem lado a lado: a imagem original e a máscara na resolução
    da malha (branco = carrinho/sólido, preto = ar/fluido), ampliada SEM
    suavizar, para você enxergar as células de verdade.
    """
    rgb, _ = _carregar_imagem(caminho_imagem, 1024)
    original = Image.fromarray(rgb)
    original = original.resize(
        (max(1, round(original.width * altura / original.height)), altura), Image.LANCZOS)

    m = np.asarray(mask, dtype=bool)
    img_mask = Image.fromarray(m.astype(np.uint8) * 255).convert("RGB")
    img_mask = img_mask.resize(
        (max(1, round(m.shape[1] * altura / m.shape[0])), altura), Image.NEAREST)

    margem, topo = 12, 22
    tela = Image.new("RGB", (original.width + img_mask.width + 3 * margem, altura + topo + margem),
                     (45, 45, 45))
    tela.paste(original, (margem, topo))
    x_mask = original.width + 2 * margem
    tela.paste(img_mask, (x_mask, topo))
    d = ImageDraw.Draw(tela)
    d.rectangle([x_mask - 1, topo - 1, x_mask + img_mask.width, topo + altura], outline=(150, 150, 150))
    d.text((margem, 5), "Imagem original", fill=(255, 255, 255))
    d.text((x_mask, 5), f"Mascara {m.shape[1]}x{m.shape[0]} (branco = carrinho)", fill=(255, 255, 255))
    return tela


def mostrar_previa(imagem, titulo="Máscara do carrinho"):
    """
    Mostra uma imagem PIL numa janela e espera você fechá-la.
    Sem tkinter, tenta o visualizador de imagens do sistema.
    Devolve True se conseguiu mostrar, False caso contrário.
    """
    try:
        import tkinter as tk
        from PIL import ImageTk
        raiz = tk.Tk()
    except Exception:
        try:
            imagem.show(title=titulo)
            return True
        except Exception:
            return False

    raiz.title(titulo)
    foto = ImageTk.PhotoImage(imagem)
    rotulo = tk.Label(raiz, image=foto)
    rotulo.image = foto                      # evita que o Python apague a imagem
    rotulo.pack()
    tk.Button(raiz, text="Fechar", command=raiz.destroy).pack(pady=4)
    _trazer_para_frente(raiz)
    raiz.mainloop()
    return True


# -------------------------------------------------------------------
# Execução:  python geometry.py              (explorador + janela de seleção)
#            python geometry.py imagem.png   (detecção automática)
#            python geometry.py imagem.png --selecionar   (com a janela)
# -------------------------------------------------------------------

def main(argv=None):
    """Fluxo de linha de comando (explorador, janela de seleção, máscara, prévia)."""
    parser = argparse.ArgumentParser(description="Gera a máscara do carrinho.")
    parser.add_argument("imagem", nargs="?",
                        help="caminho da imagem (se omitido, abre o explorador de arquivos)")
    parser.add_argument("--largura", type=int, default=400)
    parser.add_argument("--altura", type=int, default=200)
    parser.add_argument("--sem-grabcut", action="store_true")
    parser.add_argument("--sensibilidade", type=float, default=0.5)
    parser.add_argument("--modo", choices=("auto", "borda", "centro", "ia"), default="auto",
                        help="detecção: 'borda' (fundo simples), 'centro' (foto com fundo variado, "
                             "carrinho no meio), 'ia' (rede neural local, precisa do rembg) ou "
                             "'auto' (escolhe entre borda e centro)")
    parser.add_argument("--lado-segmentacao", type=int, default=512,
                        help="tamanho (px) em que a segmentação roda; maior = mais detalhe, mais lento")
    parser.add_argument("--espelhar", action="store_true")
    parser.add_argument("--saida", default="mascara.png")
    parser.add_argument("--debug", default=None, help="pasta para as etapas")
    parser.add_argument("--sem-previa", action="store_true",
                        help="não abre a janela com o resultado no final")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--selecionar", action="store_true",
                       help="abre a janela para apontar o carrinho (fundos com muita coisa)")
    grupo.add_argument("--auto", action="store_true",
                       help="só detecção automática, sem janela (mesmo escolhendo pelo explorador)")
    grupo.add_argument("--selecao", default=None, metavar="ARQ.json",
                       help="reutiliza uma seleção salva antes")
    parser.add_argument("--salvar-selecao", default=None, metavar="ARQ.json",
                        help="salva a seleção feita na janela, para reutilizar")
    a = parser.parse_args(argv)

    veio_do_explorador = not a.imagem
    caminho = a.imagem or escolher_imagem()
    if not caminho:
        raise SystemExit("Nenhuma imagem escolhida.")

    try:
        selecao = {}
        if a.selecao:
            selecao = carregar_selecao(a.selecao)
        elif a.selecionar or (veio_do_explorador and not a.auto):
            try:
                selecao = selecionar_carrinho(
                    caminho, usar_grabcut=not a.sem_grabcut, sensibilidade=a.sensibilidade,
                    modo=a.modo, lado_segmentacao=a.lado_segmentacao)
            except RuntimeError as erro:
                print(f"Aviso: sem janela de seleção ({erro}). Usando a detecção automática.")
                selecao = {}
            if selecao is None:
                raise SystemExit("Seleção cancelada.")
            if a.salvar_selecao and selecao:
                salvar_selecao(selecao, a.salvar_selecao)
                print(f"Seleção salva em {a.salvar_selecao}")

        resultado = criar_mascara(
            caminho, a.largura, a.altura,
            usar_grabcut=not a.sem_grabcut, sensibilidade=a.sensibilidade,
            espelhar_horizontal=a.espelhar, salvar_em=a.saida, debug_dir=a.debug,
            roi=selecao.get("roi"), marcas=selecao.get("marcas"), modo=a.modo,
            lado_segmentacao=a.lado_segmentacao)
    except (ValueError, OSError) as erro:
        raise SystemExit(f"Erro: {erro}")

    print(f"Imagem: {caminho}")
    print(f"Máscara salva em {os.path.abspath(a.saida)} | shape = {resultado.shape} "
          f"| células sólidas = {int(resultado.sum())}")

    if not a.sem_previa:
        previa = montar_previa(caminho, resultado)
        arquivo_previa = os.path.splitext(a.saida)[0] + "_previa.png"
        try:
            _salvar_png(arquivo_previa, np.asarray(previa))
            print(f"Prévia salva em {os.path.abspath(arquivo_previa)}")
        except OSError as erro:
            print(f"Aviso: {erro}")
        print("Abrindo a prévia... (feche a janela para terminar)")
        if not mostrar_previa(previa):
            print("Não foi possível abrir a janela; abra o arquivo da prévia "
                  f"({os.path.abspath(arquivo_previa)}) ou {os.path.abspath(a.saida)}.")


if __name__ == "__main__":
    main()
