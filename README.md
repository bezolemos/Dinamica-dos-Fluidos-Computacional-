# TrapMotion

Uma janela para simular o movimento de um carrinho de ratoeira e visualizar como o ar passa por ele. A foto fornece o perfil do carro. As medidas, a mola e o vento determinam a estimativa de alcance na pista.

## Abrir no Windows

1. Extraia **toda** a pasta do ZIP para o computador. Se já tiver clonado este repositório, abra a pasta atualizada.
2. Dê dois cliques em **Iniciar_Windows.bat**. É preciso ter Python 3.11 ou mais recente de 64 bits, com pip e Tcl/Tk. A primeira abertura usa a internet para preparar as bibliotecas na própria pasta. Nas seguintes, o programa usa essa instalação local.
3. Clique em **Simular** para experimentar a foto e os números de exemplo. Para usar o seu carrinho, clique em **Escolher foto…**, preencha as medidas e clique em **Simular** novamente. Os campos aceitam vírgula ou ponto decimal.

A entrada do programa para o VS Code é o **main.py da pasta principal**. No Windows, após a primeira abertura, selecione o interpretador `.venv\Scripts\python.exe` no VS Code. No terminal também é possível executar `.\.venv\Scripts\python.exe main.py` dentro da pasta do projeto.

Em Linux ou macOS, com Python e Tk disponíveis, abra um terminal na pasta e execute:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## Primeiro ensaio com seu carrinho

- **Foto.** Fotografe de lado, na altura das rodas, com o carro inteiro e um fundo contrastante. Escolha na tela se a frente aponta para a direita ou para a esquerda na foto. Se o fundo tiver outros objetos, arraste um retângulo na prévia ao redor do carro. Também é possível usar um PNG com fundo transparente.
- **Medidas.** Informe em centímetros os diâmetros da roda e do eixo, a corda, a haste e o comprimento total do carrinho. Meça a largura olhando o carro de frente, de um lado ao outro. A foto lateral não mostra essa medida. Informe a massa em gramas e a meta em metros.
- **Mola.** Se conhece a constante de torção, informe o valor em N·m/rad. A opção **Calcular com balança** usa a leitura em gramas, o comprimento da haste e o ângulo da medição. Meça a partir da posição relaxada da mola, com força perpendicular à ponta da haste. O ângulo de partida é o de liberação do carro.
- **Pista.** Use vento de frente em m/s. Zero simula ar parado. Em movimento, o próprio carro já sente resistência do ar. Atrito e rolamento começam com valores ilustrativos e podem ser ajustados ao material da pista.

## Ler o resultado

| Aba | O que você encontra |
| --- | --- |
| Resultado | Se atingiu a meta, quanto faltou ou quando chegou, a distância com e sem vento, o efeito em metros e em porcentagem, velocidade, dados do carrinho e sugestões para o próximo ensaio. |
| Gráficos do movimento | Distância, velocidade, aceleração e ângulo da mola ao longo do tempo. As linhas comparam ar parado e vento informado. |
| Túnel de vento | A foto usada e a silhueta no escoamento. Selecione velocidade, pressão relativa, linhas de corrente, vetores ou a própria silhueta. Após um segundo ensaio, também é possível comparar os dois perfis. |

Em ambas as abas de gráficos, **Salvar gráfico…** exporta a figura em PNG. Confira a silhueta antes de interpretar os números. Para corrigir a seleção, volte a **Dados do carrinho**, delimite o carro na foto e simule novamente.

Com a foto incluída e os valores iniciais, a estimativa é de aproximadamente **6,29 m** na referência mecânica do Projeto 15 (sem resistência do ar), **3,89 m** em ar parado e **1,99 m** com vento frontal de **1 m/s**. A diferença entre os dois últimos valores é a perda atribuída ao vento. Esses números descrevem o exemplo.

## Como os projetos se conectam

O Projeto 17 extrai a silhueta com Pillow, OpenCV e SciPy, resolve o escoamento LBM D2Q9 e estima um coeficiente de arrasto (Cd). O comprimento medido dá a escala da foto. A altura ocupada pelo perfil multiplicada pela largura medida aproxima a área frontal.

O Projeto 15 usa essa área e o Cd a cada passo de 0,005 s, tanto enquanto a mola puxa quanto quando o carro segue por inércia. Para vento frontal, a velocidade relativa é a soma da velocidade do carro e da velocidade do vento. A força contrária ao avanço vale `0,5 × densidade do ar × Cd × área frontal × velocidade relativa²`. A simulação termina quando o carro para ou ao chegar a 60 s. Se ainda estiver em movimento sem ter atingido a meta, o resultado aparece como inconclusivo.

O túnel usa uma malha fixa de 120 × 48 células, densidade do ar de referência de 1,225 kg/m³ para a conversão e escoamento 2D. Os mapas em m/s e Pa representam a solução de referência escalada para o instante de maior velocidade do carro. O teto e a base são periódicos. A simulação não reproduz o chão, a rotação das rodas nem toda a forma tridimensional do chassi. O Cd é mantido constante na corrida e o número de Reynolds da malha pode ser bem diferente do real. Use as previsões para comparar hipóteses e confira o alcance em uma pista.

## Arquivos e testes

O `main.py` inicia a janela. A pasta `trapmotion/` reúne a interface, os gráficos do movimento, o motor do Projeto 15 e a integração. A pasta `aerodinamica/` contém a geometria, o LBM, as métricas e as visualizações do Projeto 17. A foto fica em `exemplos/`. A auditoria explica o destino de cada arquivo antigo. [Organização dos arquivos](docs/Organizacao.md)

Para executar os testes na mesma instalação, na pasta principal:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Em Linux ou macOS, use `.venv/bin/python` no lugar do executável do Windows. Dois testes herdados registram limites conhecidos na segmentação de imagens, um com sombra e outro com pouco contraste.
