"""Janela única do TrapMotion. Os cálculos rodam fora da thread gráfica."""

import math
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use("Agg")  # As figuras entram na janela existente, inclusive no Windows.
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageOps, ImageTk

from . import movimento
from .graficos import COMPARACAO, MAPAS, graficos_movimento, mapa_vento
from .integracao import DadosCarro, FOTO_EXEMPLO, simular_projetos

FUNDO, PAPEL, TINTA, SUAVE = "#eef2f4", "#ffffff", "#172f3a", "#47606b"
PETROLEO, VERDE, AVISO = "#113743", "#126555", "#9b3d24"
FONTE = "Segoe UI"


def numero(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


class Foto(ttk.Frame):
    """Prévia da foto. Uma caixa desenhada no canvas orienta a segmentação."""
    def __init__(self, parent, status):
        super().__init__(parent, style="Card.TFrame", padding=16)
        self.status = status
        self.caminho = self.imagem = self.miniatura = self.roi = self.inicio = None
        self.area = (0, 0, 1, 1)
        ttk.Label(self, text="Foto de lado", style="Titulo.Card.TLabel").pack(anchor="w")
        ttk.Label(self, text="Mostre o carro inteiro, de perfil, sobre um fundo que contraste.",
                  style="Nota.Card.TLabel", wraplength=420).pack(anchor="w", pady=(2, 10))
        self.canvas = tk.Canvas(self, height=175, width=260, background="#e7edef",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", lambda _: self.desenhar())
        self.canvas.bind("<ButtonPress-1>", self.apertar)
        self.canvas.bind("<B1-Motion>", self.arrastar)
        self.canvas.bind("<ButtonRelease-1>", self.terminar)
        self.nome = ttk.Label(self, style="Card.TLabel", wraplength=400)
        self.nome.pack(anchor="w", pady=(7, 6))
        acoes = ttk.Frame(self, style="Card.TFrame")
        acoes.pack(anchor="w")
        ttk.Button(acoes, text="Escolher foto…", command=self.escolher).pack(side="left", padx=(0, 6))
        ttk.Button(acoes, text="Foto de exemplo", command=lambda: self.carregar(FOTO_EXEMPLO)).pack(side="left")
        ttk.Label(self, text="Se o fundo tiver outros objetos, arraste um retângulo na foto em volta do carrinho.",
                  wraplength=405, style="Nota.Card.TLabel").pack(anchor="w", pady=(10, 4))
        self.limpar = ttk.Button(self, text="Limpar seleção", command=self.limpar_caixa, state="disabled")
        self.limpar.pack(anchor="w")
        self.carregar(FOTO_EXEMPLO)

    def escolher(self):
        caminho = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Escolher foto de lado",
            filetypes=[("Fotos do carrinho", "*.png *.jpg *.jpeg *.jfif *.webp *.bmp *.tif *.tiff *.gif"), ("Todos os arquivos", "*")])
        if caminho:
            self.carregar(caminho)

    def carregar(self, caminho):
        try:
            with Image.open(caminho) as fonte:
                foto = ImageOps.exif_transpose(fonte).convert("RGBA")
                foto.thumbnail((1000, 650))
                fundo = Image.new("RGBA", foto.size, "white")
                fundo.alpha_composite(foto)
        except (OSError, ValueError, Image.DecompressionBombError) as erro:
            self.status(f"Não consegui abrir essa foto. Escolha um PNG ou JPG válido. {erro}", True)
            return
        self.imagem, self.caminho = fundo.convert("RGB"), Path(caminho).resolve()
        self.limpar_caixa()
        origem = "Exemplo" if self.caminho == FOTO_EXEMPLO else "Foto"
        self.nome.configure(text=f"{origem} · {self.caminho.name}")
        self.status("Foto carregada. Confira os dados e clique em Simular.")

    def limpar_caixa(self):
        self.roi = self.inicio = None
        self.limpar.configure(state="disabled")
        self.desenhar()

    def desenhar(self):
        if self.imagem is None:
            return
        w, h = max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())
        if w < 4 or h < 4:
            return
        img = self.imagem.copy()
        img.thumbnail((w - 2, h - 2), Image.Resampling.LANCZOS)
        x, y = (w - img.width) / 2, (h - img.height) / 2
        self.area = (x, y, img.width, img.height)
        self.miniatura = ImageTk.PhotoImage(img, master=self)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, image=self.miniatura, anchor="nw")
        if self.roi:
            self.retangulo(self.roi)

    def ponto(self, evento):
        x, y, w, h = self.area
        return (min(1, max(0, (evento.x - x) / w)), min(1, max(0, (evento.y - y) / h)))

    def retangulo(self, caixa):
        x, y, w, h = self.area
        a, b, c, d = caixa
        self.canvas.delete("caixa")
        self.canvas.create_rectangle(x+a*w, y+b*h, x+c*w, y+d*h, outline=VERDE, width=3, tags="caixa")

    def apertar(self, evento):
        if self.imagem:
            self.inicio = self.ponto(evento)

    def arrastar(self, evento):
        if self.inicio:
            self.retangulo((*self.inicio, *self.ponto(evento)))

    def terminar(self, evento):
        if not self.inicio:
            return
        a, b = self.inicio
        c, d = self.ponto(evento)
        self.inicio = None
        if abs(c-a) > .025 and abs(d-b) > .025:
            self.roi = (min(a, c), min(b, d), max(a, c), max(b, d))
            self.limpar.configure(state="normal")
        self.desenhar()


class Aplicativo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TrapMotion · Carrinho de ratoeira")
        largura, altura = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1140, largura-32)}x{min(790, altura-76)}")
        self.minsize(min(740, largura-32), min(490, altura-85))
        self.configure(bg=FUNDO)
        self.opcoes_visuais()
        self.campos, self.canvases, self.valores = {}, {}, {}
        self.resultado = self.anterior = None
        self.fila, self.ocupado = Queue(), False
        self.titulo_resultado = tk.StringVar()
        self.linha_status = tk.StringVar(value="Escolha uma foto e confira os dados de exemplo.")
        self.rodape()
        topo = tk.Frame(self, bg=PETROLEO, padx=22, pady=10)
        topo.pack(fill="x")
        tk.Label(topo, text="TrapMotion", bg=PETROLEO, fg="white", font=(FONTE, 20, "bold")).pack(side="left")
        tk.Label(topo, text="Carrinho de ratoeira   /   Movimento e aerodinâmica", bg=PETROLEO,
                 fg="#c7dee2", font=(FONTE, 10)).pack(side="left", padx=20)
        self.abas = ttk.Notebook(self)
        self.abas.pack(fill="both", expand=True, padx=14, pady=(12, 4))
        self.entrada, self.resumo, self.mov, self.aero = [ttk.Frame(self.abas, padding=8) for _ in range(4)]
        for aba, nome in zip((self.entrada, self.resumo, self.mov, self.aero),
                             ("Dados do carrinho", "Resultado", "Gráficos do movimento", "Túnel de vento")):
            self.abas.add(aba, text=nome, state="normal" if aba == self.entrada else "disabled")
        self.formulario()
        self.pagina_resultado()
        self.pagina_grafica(self.mov, "Movimento do carrinho", False)
        self.pagina_grafica(self.aero, "Túnel de vento", True)
        self.abas.bind("<<NotebookTabChanged>>", self.mudou_aba)

    def opcoes_visuais(self):
        e = ttk.Style(self)
        e.theme_use("clam")
        e.configure(".", font=(FONTE, 10), background=FUNDO, foreground=TINTA)
        e.configure("TNotebook", borderwidth=0, background=FUNDO)
        e.configure("TNotebook.Tab", padding=(13, 7), foreground=SUAVE)
        e.map("TNotebook.Tab", background=[("selected", PAPEL)], foreground=[("selected", TINTA), ("disabled", "#737d82")])
        e.configure("Card.TFrame", background=PAPEL)
        e.configure("Card.TLabel", background=PAPEL)
        e.configure("Nota.Card.TLabel", background=PAPEL, foreground=SUAVE, font=(FONTE, 9))
        e.configure("Titulo.Card.TLabel", background=PAPEL, font=(FONTE, 13, "bold"))
        e.configure("Titulo.TLabel", font=(FONTE, 16, "bold"))
        e.configure("TButton", padding=(9, 5))
        e.configure("Executar.TButton", background=VERDE, foreground="white", font=(FONTE, 11, "bold"), padding=(24, 7))
        e.map("Executar.TButton", background=[("disabled", "#708984"), ("active", "#0c5148")])
        e.configure("TEntry", fieldbackground="white", padding=(5, 3))

    def rodape(self):
        faixa = ttk.Frame(self, padding=(20, 6, 20, 10))
        faixa.pack(side="bottom", fill="x")
        self.executar_btn = ttk.Button(faixa, text="Simular", style="Executar.TButton", command=self.simular)
        self.executar_btn.pack(side="right")
        self.rotulo_status = ttk.Label(faixa, textvariable=self.linha_status, wraplength=580, foreground=SUAVE)
        self.rotulo_status.pack(side="left", fill="x", expand=True)

    def status(self, texto, erro=False):
        self.linha_status.set(texto)
        self.rotulo_status.configure(foreground=AVISO if erro else SUAVE)

    def formulario(self):
        # O formulário inteiro rola em telas menores; o botão Simular fica sempre à vista.
        rol = tk.Canvas(self.entrada, background=FUNDO, highlightthickness=0)
        barra = ttk.Scrollbar(self.entrada, orient="vertical", command=rol.yview)
        rol.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        rol.pack(side="left", fill="both", expand=True)
        corpo = ttk.Frame(rol)
        item = rol.create_window((0, 0), window=corpo, anchor="nw")
        corpo.bind("<Configure>", lambda _: rol.configure(scrollregion=rol.bbox("all")))
        rol.bind("<Configure>", lambda e: rol.itemconfigure(item, width=e.width))
        corpo.columnconfigure(0, weight=1, uniform="col")
        corpo.columnconfigure(1, weight=1, uniform="col")
        esquerda = ttk.Frame(corpo)
        direita = ttk.Frame(corpo)
        esquerda.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        direita.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        padrao = DadosCarro()

        def secao(lado, titulo):
            card = ttk.Frame(lado, style="Card.TFrame", padding=(15, 10))
            card.pack(fill="x", pady=(0, 8))
            card.columnconfigure(1, weight=1)
            ttk.Label(card, text=titulo, style="Titulo.Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
            return card

        def campos(card, itens):
            for linha, (chave, nome) in enumerate(itens, 1):
                self.campo(card, linha, chave, nome, getattr(padrao, chave))

        carro = secao(esquerda, "Medidas do carrinho")
        campos(carro, (("roda_cm", "Diâmetro da roda (cm)"), ("eixo_cm", "Diâmetro do eixo (cm)"),
                       ("corda_cm", "Comprimento da corda (cm)"), ("haste_cm", "Comprimento da haste (cm)"),
                       ("massa_g", "Massa total (g)"), ("comprimento_cm", "Comprimento total, de lado (cm)"),
                       ("largura_cm", "Largura vista de frente (cm)")))
        mola = secao(esquerda, "Mola e tração")
        campos(mola, (("angulo_inicial", "Ângulo inicial máximo (°)"), ("angulo_partida", "Ângulo na partida (°)")))
        self.modo_mola = tk.StringVar(value="Informar constante")
        ttk.Label(mola, text="Constante da mola", style="Card.TLabel").grid(row=3, column=0, sticky="w")
        modo = ttk.Combobox(mola, textvariable=self.modo_mola, values=("Informar constante", "Calcular com balança"),
                            state="readonly", width=19)
        modo.grid(row=3, column=1, sticky="ew", pady=2)
        modo.bind("<<ComboboxSelected>>", lambda _: self.mudar_mola())
        self.k_box = ttk.Frame(mola, style="Card.TFrame")
        self.balanca_box = ttk.Frame(mola, style="Card.TFrame")
        for box in (self.k_box, self.balanca_box):
            box.columnconfigure(1, weight=1)
        self.campo(self.k_box, 0, "k_mola", "Constante k (N·m/rad)", padrao.k_mola)
        self.campo(self.balanca_box, 0, "balanca", "Leitura da balança (g)", 213.5)
        self.campo(self.balanca_box, 1, "medicao", "Ângulo na medição (°)", 90)
        ttk.Label(self.balanca_box, style="Nota.Card.TLabel", wraplength=350,
                  text="Meça a força perpendicular à haste na ponta. Conte o ângulo a partir da mola relaxada.").grid(row=2,column=0,columnspan=2,sticky="w")
        self.mudar_mola()
        self.foto = Foto(direita, self.status)
        self.foto.pack(fill="x", pady=(0, 8))
        pista = secao(direita, "Pista e vento")
        campos(pista, (("meta_m", "Meta de distância (m)"), ("vento_ms", "Vento de frente (m/s)"),
                       ("atrito", "Atrito estático (μ)"), ("rolamento", "Resistência ao rolamento (Crr)")))
        self.frente = tk.StringVar(value=padrao.frente)
        ttk.Label(pista, text="Frente do carro na foto", style="Card.TLabel").grid(row=5,column=0,sticky="w",pady=2)
        ttk.Combobox(pista, textvariable=self.frente, values=("Esquerda", "Direita"), state="readonly",
                     width=12).grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Label(pista, style="Nota.Card.TLabel", wraplength=365,
                  text="Os números iniciais são de exemplo. Para simular ar parado, use vento igual a zero.").grid(
                      row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))

        def rolar(evento):
            if self.abas.select() == str(self.entrada) and not isinstance(evento.widget, ttk.Combobox):
                n = getattr(evento, "num", 0)
                delta = -1 if n == 4 or (n != 5 and evento.delta > 0) else 1
                rol.yview_scroll(delta * 3, "units")
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(ev, rolar, add="+")

    def campo(self, pai, linha, chave, nome, padrao):
        ttk.Label(pai, text=nome, style="Card.TLabel").grid(row=linha, column=0, sticky="w", padx=(0, 10), pady=2)
        valor = tk.StringVar(value=f"{padrao:g}".replace(".", ","))
        caixa = ttk.Entry(pai, textvariable=valor, width=10)
        caixa.grid(row=linha, column=1, sticky="ew", pady=2)
        self.campos[chave] = (valor, caixa, nome)

    def mudar_mola(self):
        self.k_box.grid_remove()
        self.balanca_box.grid_remove()
        box = self.k_box if self.modo_mola.get() == "Informar constante" else self.balanca_box
        box.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

    def ler(self, chave):
        var, entrada, nome = self.campos[chave]
        try:
            valor = float(var.get().strip().replace(",", "."))
            if math.isfinite(valor):
                return valor
        except ValueError:
            pass
        entrada.focus_set()
        raise ValueError(f'Preencha “{nome}” com um número, por exemplo 0,5.')

    def simular(self):
        if self.ocupado:
            return
        try:
            valores = {c: self.ler(c) for c in self.campos if c not in ("balanca", "medicao", "k_mola")}
            if self.modo_mola.get() == "Informar constante":
                valores["k_mola"] = self.ler("k_mola")
            else:
                forca = movimento.converter_massa_balanca_para_forca(self.ler("balanca"))
                valores["k_mola"] = movimento.calcular_constante_torsional(forca, valores["haste_cm"], self.ler("medicao"))
            dados = DadosCarro(**valores, frente=self.frente.get())
            if self.foto.caminho is None:
                raise ValueError("Escolha a foto do carrinho.")
        except ValueError as erro:
            self.abas.select(self.entrada)
            self.status(str(erro), True)
            return
        caminho, roi, cache = self.foto.caminho, self.foto.roi, self.resultado.tunel if self.resultado else None
        self.foto_em_calculo = self.foto.imagem
        self.ocupado = True
        self.executar_btn.configure(state="disabled", text="Calculando…")
        self.status("Preparando a simulação…")

        def calcular():
            try:
                ensaio = simular_projetos(caminho, dados, roi=roi, tunel_anterior=cache,
                                          progresso=lambda t: self.fila.put(("etapa", t)))
                self.fila.put(("resultado", ensaio))
            except Exception as erro:
                self.fila.put(("erro", str(erro)))
        Thread(target=calcular, daemon=True).start()
        self.after(80, self.receber)

    def receber(self):
        while True:
            try:
                tipo, valor = self.fila.get_nowait()
            except Empty:
                break
            if tipo == "etapa":
                self.status(valor)
                continue
            self.ocupado = False
            self.executar_btn.configure(state="normal", text="Simular")
            if tipo == "erro":
                self.abas.select(self.entrada)
                self.status(valor, True)
                return
            self.anterior, self.resultado = self.resultado, valor
            self.limpar_graficos()
            self.lista_mapas.configure(values=(*MAPAS, COMPARACAO) if self.anterior else MAPAS)
            self.exibir_resultado()
            for aba in (self.resumo, self.mov, self.aero):
                self.abas.tab(aba, state="normal")
            self.abas.select(self.resumo)
            self.status(f"Ensaio concluído · {valor.foto.name}. Você pode mudar as medidas ou a foto e simular novamente.")
            return
        if self.ocupado:
            self.after(80, self.receber)

    def pagina_resultado(self):
        ttk.Label(self.resumo, textvariable=self.titulo_resultado, style="Titulo.TLabel", wraplength=760).pack(anchor="w", pady=(3, 8))
        cards = ttk.Frame(self.resumo)
        cards.pack(fill="x", pady=(0, 6))
        for i, nome in enumerate(("Alcance com vento", "Alcance em ar parado", "Perda pelo vento")):
            cards.columnconfigure(i, weight=1, uniform="medidas")
            card = ttk.Frame(cards, style="Card.TFrame", padding=(12, 7))
            card.grid(row=0, column=i, padx=(0, 5), sticky="nsew")
            ttk.Label(card, text=nome, style="Nota.Card.TLabel").pack(anchor="w")
            var = self.valores[nome] = tk.StringVar()
            ttk.Label(card, textvariable=var, style="Titulo.Card.TLabel").pack(anchor="w")
        painel = ttk.Frame(self.resumo)
        painel.pack(fill="both", expand=True, pady=(3, 0))
        self.relato = tk.Text(painel, wrap="word", font=(FONTE, 10), bg=PAPEL, fg=TINTA,
                              relief="flat", padx=15, pady=8, spacing3=3, height=5, width=35)
        barra = ttk.Scrollbar(painel, orient="vertical", command=self.relato.yview)
        self.relato.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.relato.pack(side="left", fill="both", expand=True)
        self.relato.tag_configure("secao", foreground=VERDE, font=(FONTE, 11, "bold"), spacing1=6)
        self.relato.configure(state="disabled")

    def exibir_resultado(self):
        r = self.resultado
        carro, corrida = r.carro, r.com_vento
        mov, meta, met = corrida["movimento"], corrida["meta"], r.tunel.metricas
        alvo = numero(carro.meta_m)
        status = {"meta_atingida": f"Meta de {alvo} m atingida em {numero(meta['tempo_meta_atingida_s'] or 0)} s",
                  "meta_nao_atingida": f"Faltaram {numero(-meta['diferenca_m'])} m para a meta de {alvo} m",
                  "resultado_inconclusivo": f"Tempo esgotado antes de alcançar a meta de {alvo} m"}
        self.titulo_resultado.set(status[meta["resultado"]])
        def metros(v):
            return numero(v) + " m"
        for nome, valor in (("Alcance com vento", mov["distancia_total_m"]),
                            ("Alcance em ar parado", r.ar_parado["movimento"]["distancia_total_m"]),
                            ("Perda pelo vento", r.perda_m)):
            self.valores[nome].set(metros(valor))
        t = self.relato
        t.configure(state="normal")
        t.delete("1.0", "end")

        def escrever(titulo, linhas):
            t.insert("end", titulo + "\n", "secao")
            for linha in linhas:
                t.insert("end", linha + "\n")

        delta = ("O carrinho ficou parado em ar parado. A perda percentual não pode ser calculada."
                 if r.perda_pct is None else
                 f"Com vento frontal de {numero(carro.vento_ms)} m/s, a distância caiu {numero(r.perda_pct, 1)}% em relação ao ar parado."
                 if r.perda_m > .0005 else "O vento informado não alterou o alcance na precisão exibida.")
        escrever("O que aconteceu na corrida", [
            delta,
            f"Velocidade máxima de {numero(mov['velocidade_maxima_m_s'])} m/s em uma corrida de {numero(mov['tempo_total_s'])} s. "
            f"A mola impulsionou o carro por {metros(mov['distancia_fase_impulsionada_m'])} e ele percorreu {metros(mov['distancia_fase_livre_m'])} por inércia.",
            f"Referência do Projeto 15 sem resistência do ar: {metros(r.mecanico['movimento']['distancia_total_m'])}."])
        if any(e["movimento"]["motivo_encerramento"] == "tempo_maximo_atingido" for e in (r.ar_parado, r.com_vento, r.mecanico)):
            escrever("Limite de tempo", ["Pelo menos uma corrida ainda estava em movimento aos 60 s. Os alcances e diferenças comparam a distância observada até esse limite."])
        escrever("O que testar no carrinho", [f"•  {rec}" for rec in r.recomendacoes])
        if corrida["diagnosticos"]:
            escrever("Diagnóstico mecânico", corrida["diagnosticos"])
        escrever("Dados deste ensaio", [
            f"{r.foto.name}  ·  {numero(carro.comprimento_cm, 1)} × {numero(carro.largura_cm, 1)} cm (comprimento × largura)  ·  {numero(carro.massa_g, 0)} g.",
            f"Roda {numero(carro.roda_cm)} cm  ·  eixo {numero(carro.eixo_cm)} cm  ·  corda {numero(carro.corda_cm)} cm  ·  haste {numero(carro.haste_cm)} cm.",
            f"Constante da mola {numero(carro.k_mola, 3)} N·m/rad  ·  ângulo inicial {numero(carro.angulo_inicial, 0)}°  ·  partida {numero(carro.angulo_partida, 0)}°.",
            f"Área frontal estimada {numero(r.area_m2*10000, 1)} cm²  ·  coeficiente de arrasto 2D {numero(met['coeficiente_arrasto'])}  ·  "
            f"arrasto máximo estimado {numero(mov['arrasto_maximo_n'], 3)} N.",
            f"Túnel 2D: Reynolds da malha {numero(met['reynolds'], 1)} e {r.tunel.iteracoes} passos. O arrasto do carro real precisa ser conferido na pista."])
        if r.tunel.avisos:
            escrever("Observações sobre o túnel e a silhueta", r.tunel.avisos)
        t.configure(state="disabled")
        t.yview_moveto(0)
        mini = self.foto_em_calculo.copy()
        mini.thumbnail((155, 55), Image.Resampling.LANCZOS)
        self.foto_resultado = ImageTk.PhotoImage(mini, master=self)
        self.foto_tunel.configure(image=self.foto_resultado)
        self.legenda_fluxo = (
            f"Ar da esquerda para a direita. Mapas estimados no pico da corrida, com velocidade do ar em relação ao carro de "
            f"{numero(r.velocidade_relativa_ms)} m/s (vento + velocidade do carro).")

    def pagina_grafica(self, aba, titulo, aero):
        cab = ttk.Frame(aba)
        cab.pack(fill="x", pady=(2, 5))
        ttk.Label(cab, text=titulo, style="Titulo.TLabel").pack(side="left")
        ttk.Button(cab, text="Salvar gráfico…", command=lambda: self.salvar(aba)).pack(side="right")
        if aero:
            self.tipo_mapa = tk.StringVar(value=MAPAS[0])
            self.lista_mapas = ttk.Combobox(cab, textvariable=self.tipo_mapa, state="readonly", width=21, values=MAPAS)
            self.lista_mapas.pack(side="right", padx=8)
            self.lista_mapas.bind("<<ComboboxSelected>>", lambda _: self.desenhar_grafico(aba))
            self.nota_aero = tk.StringVar()
            faixa = ttk.Frame(aba)
            faixa.pack(fill="x", pady=(0, 4))
            self.foto_tunel = ttk.Label(faixa)
            self.foto_tunel.pack(side="left", padx=(0, 12))
            ttk.Label(faixa, textvariable=self.nota_aero, wraplength=560).pack(side="left", anchor="center")
        container = ttk.Frame(aba, style="Card.TFrame")
        container.pack(fill="both", expand=True)
        container.pack_propagate(False)
        aba.container = container

    def descartar(self, aba):
        anterior = self.canvases.pop(aba, None)
        if anterior:
            anterior.get_tk_widget().destroy()
            plt.close(anterior.figure)

    def limpar_graficos(self):
        for aba in list(self.canvases):
            self.descartar(aba)

    def mudou_aba(self, _=None):
        if self.resultado is None:
            return
        aba = self.nametowidget(self.abas.select())
        if aba in (self.mov, self.aero) and aba not in self.canvases:
            self.desenhar_grafico(aba)

    def desenhar_grafico(self, aba):
        if not self.resultado:
            return
        self.descartar(aba)
        if aba == self.aero:
            tipo = self.tipo_mapa.get()
            descricao = self.legenda_fluxo
            if tipo == COMPARACAO:
                descricao = "Silhuetas do ensaio anterior e do atual, ajustadas à mesma malha para comparar a forma."
            elif tipo == "Silhueta":
                descricao = "Confira se a área sólida corresponde ao perfil da foto. Se necessário, volte aos dados e delimite o carro com o mouse."
            self.nota_aero.set(descricao)
            fig = mapa_vento(self.resultado, tipo, self.anterior)
        else:
            fig = graficos_movimento(self.resultado)
        canvas = FigureCanvasTkAgg(fig, master=aba.container)
        self.canvases[aba] = canvas
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

    def salvar(self, aba):
        if aba not in self.canvases:
            return
        caminho = filedialog.asksaveasfilename(parent=self, title="Salvar gráfico", defaultextension=".png",
                  initialfile="tunel_de_vento.png" if aba == self.aero else "movimento.png", filetypes=[("Imagem PNG", "*.png")])
        if caminho:
            try:
                self.canvases[aba].figure.savefig(caminho, dpi=180, bbox_inches="tight")
                self.status(f"Gráfico salvo em {caminho}")
            except OSError as erro:
                self.status(f"Não foi possível salvar o gráfico. {erro}", True)

    def destroy(self):
        self.limpar_graficos()
        super().destroy()


def iniciar():
    Aplicativo().mainloop()
