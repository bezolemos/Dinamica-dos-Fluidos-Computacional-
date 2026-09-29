# Projeto 17 — Guia de Trabalho da Equipe

## Objetivo do projeto

Desenvolver, em Python, um **simulador aerodinâmico 2D de nível intermediário**, usando uma abordagem CFD simplificada baseada em **Lattice Boltzmann Method (LBM), modelo D2Q9**.

O programa deverá receber a **silhueta 2D de um carrinho**, simular o fluxo de ar ao redor dela e gerar resultados que possam ser analisados e visualizados.

> Cada integrante será responsável principalmente por um arquivo.  
> A ideia deste documento **não é entregar o código pronto**, mas dizer o que cada pessoa precisa estudar, pesquisar e desenvolver.

---

# 1. Organização dos arquivos

```text
projeto17/
│
├── main.py
├── geometry.py
├── solver_lbm.py
├── metrics.py
├── visualization.py
│
├── examples/
├── tests/
├── requirements.txt
└── README.md
```

Fluxo principal:

```text
geometry.py
    ↓
máscara do carrinho
    ↓
solver_lbm.py
    ↓
rho, ux, uy
    ↓
 ┌───────────────┐
 ↓               ↓
metrics.py   visualization.py
 ↓               ↓
números         gráficos
    \             /
     \           /
        main.py
```

---

# 2. `geometry.py` — Geometria do carrinho

## Objetivo

Transformar uma imagem/silhueta 2D do carrinho em uma estrutura que o simulador consiga entender.

O solver não deve receber uma imagem comum. Ele deve receber uma **máscara binária** indicando quais pontos da malha pertencem ao carrinho e quais pertencem ao ar.

Exemplo conceitual:

```text
0 0 0 0 0 0
0 0 1 1 0 0
0 1 1 1 1 0
0 0 1 1 0 0
```

Onde:

- `0` / `False` = espaço onde existe ar;
- `1` / `True` = região ocupada pelo carrinho.

## O que estudar/pesquisar

Pesquisar:

- como abrir imagens com **Pillow**;
- como converter uma imagem para escala de cinza;
- como transformar uma imagem em uma matriz **NumPy**;
- o que é **threshold** ou limiar de imagem;
- como redimensionar uma imagem;
- como criar uma matriz booleana (`True` / `False`);
- noções básicas de processamento de imagem;
- opcionalmente, ferramentas do `scipy.ndimage` para limpar pequenas imperfeições da máscara.

## O que programar

O arquivo deverá ser capaz de:

1. receber o caminho de uma imagem;
2. abrir essa imagem;
3. converter a imagem para um formato simples;
4. identificar o que é carrinho e o que é fundo;
5. transformar isso em uma máscara booleana;
6. redimensionar a máscara para caber na malha da simulação;
7. retornar essa máscara para o restante do projeto.

## Entrada esperada

Exemplo conceitual:

```text
caminho da imagem
largura da malha
altura da malha
```

## Saída esperada

```text
mask
```

`mask` deve ser uma matriz NumPy de valores `True` e `False`.

## Bibliotecas principais

```text
Pillow
NumPy
SciPy (opcional)
```

## Fórmulas

Nesta parte não há uma fórmula física principal.

O foco é **tratamento de imagem e geometria**.

---

# 3. `solver_lbm.py` — Simulação CFD

## Objetivo

Este arquivo será o **motor físico do projeto**.

Ele deverá simular o movimento do ar dentro de uma malha 2D usando o **Lattice Boltzmann Method — D2Q9**.

O solver deverá receber a máscara produzida por `geometry.py` e calcular o fluxo de ar ao redor do carrinho.

## O que estudar/pesquisar

Pesquisar:

- o que é CFD;
- o que é Lattice Boltzmann Method;
- modelo **D2Q9**;
- lattice/grid 2D;
- distribuições `f0 ... f8`;
- densidade do fluido;
- velocidade macroscópica;
- distribuição de equilíbrio;
- etapa de colisão;
- etapa de streaming;
- parâmetro de relaxamento `tau`;
- viscosidade no LBM;
- condição de contorno **bounce-back**;
- entrada de fluxo constante;
- funcionamento básico do NumPy com matrizes multidimensionais.

## Conceito do D2Q9

Cada célula possui nove possíveis direções:

```text
↖ ↑ ↗
← • →
↙ ↓ ↘
```

Uma fica parada no centro e oito representam movimento para os lados e diagonais.

## Fórmulas principais

### Densidade

\[
\rho = \sum_{i=0}^{8} f_i
\]

A densidade é obtida somando as nove distribuições de uma célula.

---

### Velocidade do fluido

\[
\vec{u} =
\frac{1}{\rho}
\sum_i f_i \vec{e_i}
\]

O resultado deverá gerar duas componentes:

```text
ux = velocidade horizontal
uy = velocidade vertical
```

---

### Pesos D2Q9

Centro:

\[
w_0 = \frac{4}{9}
\]

Direções horizontais e verticais:

\[
w_{1-4} = \frac{1}{9}
\]

Diagonais:

\[
w_{5-8} = \frac{1}{36}
\]

---

### Distribuição de equilíbrio

\[
f_i^{eq}
=
w_i\rho
\left[
1
+3(\vec{e_i}\cdot\vec{u})
+\frac{9}{2}(\vec{e_i}\cdot\vec{u})^2
-\frac{3}{2}|\vec{u}|^2
\right]
\]

Pesquisar o significado de cada termo antes de implementar.

---

### Colisão

\[
f_i^*
=
f_i
-
\frac{1}{\tau}
(f_i-f_i^{eq})
\]

Essa etapa aproxima o fluido do estado de equilíbrio.

---

### Streaming

\[
f_i(x+e_i,t+1)=f_i^*(x,t)
\]

Depois da colisão, cada distribuição avança para a célula vizinha correspondente à sua direção.

---

### Viscosidade em unidades de lattice

\[
\nu =
\frac{\tau-0.5}{3}
\]

ou:

\[
\tau = 0.5 + 3\nu
\]

---

## Bounce-back

Quando o fluxo encontra uma célula que pertence ao carrinho, ele não pode atravessá-la.

A distribuição deverá ser refletida para a direção oposta.

Pesquisar:

```text
LBM bounce-back boundary condition
```

## O que programar

O arquivo deverá, progressivamente:

1. criar a estrutura D2Q9;
2. definir as nove direções;
3. definir os pesos;
4. inicializar densidade e velocidade;
5. inicializar as distribuições;
6. calcular densidade;
7. calcular `ux` e `uy`;
8. calcular o equilíbrio;
9. realizar colisão;
10. realizar streaming;
11. aplicar bounce-back usando `mask`;
12. criar uma entrada de vento;
13. repetir o processo durante várias iterações;
14. retornar os resultados finais.

## Entrada esperada

```text
mask
velocidade do vento
número de iterações
tau / viscosidade
```

## Saída esperada

```text
rho
ux
uy
```

## Bibliotecas principais

```text
NumPy
SciPy, se realmente necessário
```

---

# 4. `metrics.py` — Resultados aerodinâmicos

## Objetivo

Transformar os dados brutos do solver em **informações físicas compreensíveis**.

Esse arquivo não deve executar a simulação CFD.

Ele recebe os resultados já calculados por `solver_lbm.py` e calcula métricas.

## O que estudar/pesquisar

Pesquisar:

- magnitude de um vetor;
- velocidade do escoamento;
- pressão no LBM;
- número de Reynolds;
- viscosidade cinemática;
- força de arrasto;
- coeficiente de arrasto;
- diferença entre `ux`, `uy` e velocidade resultante;
- quais resultados são úteis para comparar dois chassis.

## Fórmulas principais

### Velocidade resultante

\[
v = \sqrt{u_x^2 + u_y^2}
\]

Essa fórmula transforma as duas componentes de velocidade em uma velocidade total.

---

### Pressão no D2Q9

\[
p = c_s^2\rho
\]

Para o D2Q9:

\[
c_s^2 = \frac{1}{3}
\]

Portanto:

\[
p = \frac{\rho}{3}
\]

---

### Número de Reynolds

\[
Re = \frac{UL}{\nu}
\]

Onde:

- `U` = velocidade característica do fluxo;
- `L` = comprimento característico do carrinho;
- `ν` = viscosidade cinemática.

Pesquisar o significado físico do número de Reynolds.

---

### Força de arrasto

Quando for possível obter ou estimar o coeficiente de arrasto:

\[
F_d =
\frac{1}{2}
\rho U^2 A C_d
\]

Onde:

- `Fd` = força de arrasto;
- `ρ` = densidade do ar;
- `U` = velocidade;
- `A` = área de referência;
- `Cd` = coeficiente de arrasto.

---

### Coeficiente de arrasto

\[
C_d =
\frac{F_d}
{\frac{1}{2}\rho U^2 A}
\]

A equipe deverá pesquisar qual método será usado para estimar `Fd` ou `Cd` a partir do CFD.

## O que programar

Criar funções independentes para:

1. calcular velocidade resultante;
2. encontrar velocidade máxima;
3. calcular velocidade média;
4. calcular pressão;
5. calcular Reynolds;
6. calcular/estimar força de arrasto;
7. calcular/estimar coeficiente de arrasto;
8. organizar os resultados de forma simples para o `main.py`.

## Entrada esperada

```text
rho
ux
uy
mask
parâmetros físicos
```

## Saída esperada

Exemplo conceitual:

```text
velocidade máxima
velocidade média
Reynolds
pressão
força de arrasto
coeficiente de arrasto
```

## Bibliotecas principais

```text
NumPy
math
```

---

# 5. `visualization.py` — Visualização do túnel de vento

## Objetivo

Transformar os resultados numéricos em imagens e gráficos que permitam entender o comportamento do ar.

Esse arquivo não deve calcular o CFD.

Ele recebe os resultados do solver e apenas os representa visualmente.

## O que estudar/pesquisar

Pesquisar no **Matplotlib**:

- `imshow`;
- `streamplot`;
- `quiver`;
- `colorbar`;
- criação de figuras;
- criação de títulos e eixos;
- como sobrepor a máscara do carrinho ao gráfico;
- como salvar gráficos como imagem.

Também pesquisar:

- o que é um campo vetorial;
- o que são linhas de corrente;
- como representar magnitude de velocidade usando cores.

## O que programar

Criar funções para:

1. mostrar a silhueta do carrinho na malha;
2. mostrar o mapa de velocidade;
3. mostrar o mapa de pressão;
4. mostrar as linhas de corrente;
5. mostrar vetores de velocidade;
6. destacar a região de esteira atrás do carrinho;
7. salvar gráficos em arquivo;
8. permitir comparação visual entre dois chassis.

## Entrada esperada

```text
rho
ux
uy
mask
```

## Saída esperada

Gráficos como:

```text
mapa de velocidade
mapa de pressão
linhas de fluxo
campo vetorial
```

## Bibliotecas principais

```text
Matplotlib
NumPy
```

## Fórmulas

O ideal é receber do `metrics.py` os valores físicos já calculados.

Para testes independentes, pode usar a magnitude:

\[
v = \sqrt{u_x^2 + u_y^2}
\]

---

# 6. `main.py` — Integração

## Responsável principal

Bernardo.

## Objetivo

O `main.py` não deve conter toda a física.

Ele deve apenas **organizar a execução dos outros módulos**.

Fluxo esperado:

```text
1. carregar geometria
2. gerar máscara
3. executar CFD
4. calcular métricas
5. gerar visualizações
6. apresentar resultados
```

O `main.py` será desenvolvido principalmente quando os outros módulos já estiverem funcionando.

---

# 7. Bibliotecas do projeto

Inicialmente:

```text
numpy
scipy
matplotlib
pillow
pytest
```

Instalação:

```bash
pip install numpy scipy matplotlib pillow pytest
```

---

# 8. Regras para trabalhar no GitHub

Cada integrante deve trabalhar principalmente no arquivo pelo qual é responsável.

Sugestão de branches:

```text
feature-geometry
feature-solver
feature-metrics
feature-visualization
```

Fluxo:

```text
criar/entrar na branch
        ↓
programar
        ↓
testar
        ↓
git add
        ↓
git commit
        ↓
git push
        ↓
Pull Request
        ↓
revisão
        ↓
merge na main
```

Os commits devem explicar claramente a mudança.

Exemplos:

```text
feat: add image mask conversion
feat: add D2Q9 lattice initialization
feat: add Reynolds calculation
feat: add velocity field visualization
fix: correct image resizing
test: add geometry mask tests
```

Evitar mensagens como:

```text
update
mudanças
teste
coisas novas
```

---

# 9. Regra mais importante da equipe

Cada módulo deve respeitar suas entradas e saídas.

```text
geometry.py
    ↓
mask

solver_lbm.py
    ↓
rho, ux, uy

metrics.py
    ↓
resultados físicos

visualization.py
    ↓
gráficos
```

Isso permitirá desenvolver as partes separadamente e depois conectá-las sem precisar reescrever todo o projeto.

---

# 10. Como cada integrante deve começar

## Geometria

Primeiro objetivo:

> Conseguir abrir uma imagem simples em preto e branco e transformá-la em uma matriz `True/False`.

Ainda não precisa usar o carrinho definitivo.

---

## Solver

Primeiro objetivo:

> Criar a malha D2Q9, suas nove direções e seus pesos.

Ainda não tentar programar toda a simulação.

---

## Métricas

Primeiro objetivo:

> Criar e testar os cálculos de velocidade resultante, pressão e Reynolds usando valores fictícios.

Não precisa esperar o solver ficar pronto.

---

## Visualização

Primeiro objetivo:

> Criar matrizes falsas de `ux` e `uy` e aprender a mostrá-las usando `imshow`, `streamplot` e `quiver`.

Não precisa esperar o CFD funcionar.

---

# Resultado esperado

No final, cada módulo poderá ser desenvolvido e testado separadamente.

Quando os quatro estiverem funcionando:

```text
GEOMETRIA
    ↓
CFD
    ↓
MÉTRICAS
    ↓
VISUALIZAÇÃO
```

Depois disso, o Projeto 17 poderá ser preparado para integração futura com o **Projeto 15**.
