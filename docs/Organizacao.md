# Destino dos arquivos originais

A revisão partiu dos 29 arquivos da versão do Projeto 17 registrada no commit `5e85ae5`. Os exemplos sintéticos de gráficos podem ser reproduzidos pelas funções de visualização. A imagem real do projeto foi mantida para que qualquer integrante consiga testar a instalação. [Versão anterior do Projeto 17](https://github.com/bezolemos/Dinamica-dos-Fluidos-Computacional-/tree/5e85ae53af68223eef045018f00f2ac8561325e9)

| Arquivo ou grupo da versão anterior | Qtde. | Destino |
| --- | ---: | --- |
| `CFD/main.py` e `CFD/visualizar.py` | 2 | Ambos estavam vazios. A entrada é agora `main.py` na raiz. Os gráficos vêm do módulo completo abaixo. |
| `CFD/geometry.py` e `CFD/geometry.pyTeste/Geometry.py` | 2 | Versões intermediárias, substituídas pela geometria V2. |
| `CFD/geometry.py V2/geometry.py` e `test_geometry.py` na mesma pasta | 2 | Preservados em `aerodinamica/geometry.py` e `tests/test_geometry.py`. Os testes importam o mesmo módulo que a janela. |
| `mascara.png` e `mascara_previa.png` dentro da geometria V2 | 2 | Prévias de uma execução antiga. A janela mostra a silhueta da foto atual. |
| `CFD/LBM.py` e `CFD/metrics.py` | 2 | Movidos para `aerodinamica/lbm.py` e `aerodinamica/metrics.py`. O solver pode informar o andamento à janela. As métricas verificam os dois componentes da velocidade. |
| `CFD/tests/test_lbm.py` e `test_metrics.py` | 2 | Movidos para `tests/`, com imports pelo pacote. |
| `CFD/visualization.py/visualization.py` | 1 | Implementação inicial, substituída pela V2. |
| Os PNGs em `CFD/visualization.py/` | 5 | Resultados de demonstração retirados da pasta de código. |
| `CFD/visualization.pyv2/visualization.py` | 1 | Preservado em `aerodinamica/visualizar.py`. Recebe os campos calculados e permite indicar as unidades dos mapas. |
| Os PNGs em `CFD/visualization.pyv2/resultados_teste/` | 8 | Imagens pré-geradas retiradas. É possível salvar os novos resultados pela janela. |
| `mousetrap car imagem lado.png` | 1 | `exemplos/carrinho.png`, usada no ensaio de demonstração. |
| `CFD/Projeto17_Guia_Equipe.md` | 1 | `docs/Guia_original_da_equipe.md`, preservado como registro da divisão inicial de trabalho. |

## Projeto 15 e integração

O módulo `simulacao.py` do TrapMotion foi incorporado como `trapmotion/movimento.py`. Os cálculos da mola, da transmissão, da aderência, do rolamento e dos históricos vêm da implementação do grupo. A resistência aerodinâmica passa a atuar nas duas fases do movimento. A configuração com Cd × área igual a zero preserva o cálculo mecânico original. Os quatro gráficos da interface V11 foram adaptados para comparar as duas corridas. [Código de origem do Projeto 15](https://github.com/bezolemos/TrapMotion/tree/dbe28d7365927b8f72347a7fb786dcea60f71af9)

A ligação entre a foto, o arrasto e o movimento fica em `trapmotion/integracao.py`. A janela única fica em `trapmotion/interface.py`. Para começar, abra o iniciador da pasta principal e use o exemplo. [Instruções de uso](../README.md)
