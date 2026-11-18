import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DADOS FICTÍCIOS PARA TESTE DA VISUALIZAÇÃO
# ============================================================

ux = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

uy = np.array([
    [0,  0,  0,  0,  0],
    [1,  1,  0, -1, -1],
    [2,  1,  0, -1, -2],
    [1,  1,  0, -1, -1],
    [0,  0,  0,  0,  0]
])


# ============================================================
# CÁLCULOS DOS DADOS DE TESTE
# ============================================================

# Magnitude da velocidade
velocidade = np.sqrt(ux**2 + uy**2)

# Densidade fictícia
rho = np.ones_like(velocidade)

# Pressão fictícia baseada na velocidade
pressao = 1 - 0.1 * velocidade


# ============================================================
# MÁSCARA DO CARRO
# ============================================================

mask = np.zeros_like(velocidade)

# Representação do chassi
mask[2:4, 1:4] = 1


# Segundo chassi para comparação
mask_chassi_2 = np.zeros_like(velocidade)
mask_chassi_2[1:3, 1:4] = 1


# ============================================================
# CAMPO DE VELOCIDADE
# ============================================================

def mostrar_velocidade(velocidade, ux, uy, mask):

    X, Y = np.meshgrid(
        np.arange(5),
        np.arange(5)
    )

    plt.figure(figsize=(10, 6))

    # Mapa de velocidade
    imagem = plt.imshow(
        velocidade,
        cmap="viridis"
    )

    plt.colorbar(
        imagem,
        label="Velocidade"
    )

    # Contorno do carro
    plt.contour(
        mask,
        levels=[0.5],
        colors="black"
    )

    # Linhas de fluxo
    plt.streamplot(
        X,
        Y,
        ux,
        uy,
        color="white"
    )

    # Vetores de velocidade
    plt.quiver(
        X,
        Y,
        ux,
        uy
    )

    # Preenchimento da máscara
    plt.contourf(
        mask,
        levels=[0.5, 1],
        colors="gray",
        alpha=0.8
    )

    # Região de wake
    plt.axvspan(
        4,
        5,
        color="red",
        alpha=0.15
    )

    plt.text(
        4.1,
        4.5,
        "Wake",
        color="red"
    )

    plt.title("Campo de velocidade")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Salva a imagem
    plt.savefig("campo_velocidade.png")


mostrar_velocidade(
    velocidade,
    ux,
    uy,
    mask
)


# ============================================================
# CAMPO DE PRESSÃO
# ============================================================

def mostrar_pressao(pressao, mask):

    plt.figure(figsize=(10, 6))

    # Mapa de pressão
    imagem = plt.imshow(
        pressao,
        cmap="plasma"
    )

    plt.colorbar(
        imagem,
        label="Pressão"
    )

    # Contorno do carro
    plt.contour(
        mask,
        levels=[0.5],
        colors="black"
    )

    # Preenchimento do carro
    plt.contourf(
        mask,
        levels=[0.5, 1],
        colors="gray",
        alpha=0.8
    )

    plt.title("Campo de Pressão")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Salva a imagem
    plt.savefig("campo_pressao.png")


mostrar_pressao(
    pressao,
    mask
)


# ============================================================
# LINHAS DE FLUXO
# ============================================================

def mostrar_linhas_fluxo(ux, uy, mask):

    X, Y = np.meshgrid(
        np.arange(5),
        np.arange(5)
    )

    plt.figure(figsize=(10, 6))

    # Linhas de fluxo
    plt.streamplot(
        X,
        Y,
        ux,
        uy,
        color="black"
    )

    # Contorno do carro
    plt.contour(
        mask,
        levels=[0.5],
        colors="red"
    )

    # Preenchimento do carro
    plt.contourf(
        mask,
        levels=[0.5, 1],
        colors="gray",
        alpha=0.8
    )

    plt.title("Linhas de Fluxo")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Salva a imagem
    plt.savefig("linhas_fluxo.png")


mostrar_linhas_fluxo(
    ux,
    uy,
    mask
)


# ============================================================
# VETORES DE VELOCIDADE
# ============================================================

def mostrar_vetores(ux, uy, mask):

    X, Y = np.meshgrid(
        np.arange(5),
        np.arange(5)
    )

    plt.figure(figsize=(10, 6))

    # Vetores de velocidade
    plt.quiver(
        X,
        Y,
        ux,
        uy
    )

    # Contorno do carro
    plt.contour(
        mask,
        levels=[0.5],
        colors="red"
    )

    # Preenchimento do carro
    plt.contourf(
        mask,
        levels=[0.5, 1],
        colors="gray",
        alpha=0.8
    )

    plt.title("Vetores de Velocidade")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Salva a imagem
    plt.savefig("vetores_velocidade.png")


mostrar_vetores(
    ux,
    uy,
    mask
)

# ============================================================
# COMPARAÇÃO ENTRE DOIS CHASSIS
# ============================================================

def comparar_chassis(mask_chassi_1, mask_chassi_2):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(mask_chassi_1, cmap="gray")
    plt.title("Chassi 1")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_chassi_2, cmap="gray")
    plt.title("Chassi 2")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()

    plt.savefig("comparacao_chassis.png")

comparar_chassis(
    mask,
    mask_chassi_2
)


# ============================================================
# EXIBIR TODOS OS GRÁFICOS
# ============================================================

plt.show()