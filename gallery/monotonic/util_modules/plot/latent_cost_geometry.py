import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Dim reduction helpers
# -----------------------------
def embed(Z: np.ndarray, method: str = "pca", n_components: int = 2, seed: int = 0, **kwargs) -> np.ndarray:
    """
    Z: (N, 64) latent vectors
    returns: (N, n_components) embedding
    method: "pca" | "tsne" | "umap"
    """
    Z = np.asarray(Z)
    assert Z.ndim == 2 and Z.shape[0] >= 2, "Z should be (N, D), N>=2"
    assert n_components in (2, 3)

    method = method.lower()
    if method == "pca":
        model = PCA(n_components=n_components, random_state=seed)
        return model.fit_transform(Z)

    if method == "tsne":
        # t-SNE는 느리고 파라미터 민감함. 기본값은 무난한 편.
        model = TSNE(
            n_components=n_components,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            **kwargs
        )
        return model.fit_transform(Z)

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("UMAP을 쓰려면 `pip install umap-learn` 필요") from e
        model = umap.UMAP(n_components=n_components, random_state=seed, **kwargs)
        return model.fit_transform(Z)

    raise ValueError(f"Unknown method: {method}")

# -----------------------------
# 2D plot: (x,y) with cost as color
# -----------------------------
def plot_2d(Z: np.ndarray, cost: np.ndarray, method: str = "pca", type:str = "latent", seed: int = 0, s: int = 12, alpha: float = 0.8, save_path: str = None, show: bool = True, **embed_kwargs):
    """
    2D embedding scatter, color = cost
    """
    cost = np.asarray(cost).reshape(-1)
    assert len(cost) == Z.shape[0], "cost length must match Z rows"

    E = embed(Z, method=method, n_components=2, seed=seed, **embed_kwargs)
    x, y = E[:, 0], E[:, 1]

    plt.figure()
    sc = plt.scatter(x, y, c=cost, s=s, alpha=alpha, cmap="viridis_r")
    plt.colorbar(sc, label="cost")
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")
    plt.title(f"{type.capitalize()} ({Z.shape[-1]}D) -> 2D via {method.upper()} (color=cost)")
    plt.tight_layout()

    fig = plt.gcf()

    if show:
        plt.show()
    if save_path is not None:
        return fig
    

# -----------------------------
# 3D plot A: (x,y,z=cost) + color=cost
# -----------------------------
def plot_3d_height(Z: np.ndarray, cost: np.ndarray, method: str = "pca", seed: int = 0, s: int = 10, alpha: float = 0.75, save_path: str = None, show: bool = True, **embed_kwargs):
    """
    2D embedding provides x,y; z-axis is cost (height plot).
    Good when you literally want "graph of cost function" style.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cost = np.asarray(cost).reshape(-1)
    assert len(cost) == Z.shape[0], "cost length must match Z rows"

    E = embed(Z, method=method, n_components=2, seed=seed, **embed_kwargs)
    x, y = E[:, 0], E[:, 1]
    z = cost

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=cost, s=s, alpha=alpha, cmap="viridis_r")
    fig.colorbar(sc, ax=ax, label="cost", shrink=0.7)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_zlabel("cost")
    ax.set_title(f"3D: x,y from {method.upper()}-2D; z=cost")
    plt.tight_layout()

    fig = plt.gcf()
    
    if show:
        plt.show()
    if save_path is not None:
        return fig

# -----------------------------
# 3D plot B: (x,y,z) = 3D embedding, color=cost
# -----------------------------
def plot_3d_embed(Z: np.ndarray, cost: np.ndarray, method: str = "pca", type:str = "latent", seed: int = 0, s: int = 10, alpha: float = 0.75, save_path: str = None, show: bool = True, **embed_kwargs):
    """
    3D embedding scatter. Axes are latent embedding dimensions; color=cost.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cost = np.asarray(cost).reshape(-1)
    assert len(cost) == Z.shape[0], "cost length must match Z rows"

    E = embed(Z, method=method, n_components=3, seed=seed, **embed_kwargs)
    x, y, z = E[:, 0], E[:, 1], E[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=cost, s=s, alpha=alpha, cmap="viridis_r")
    fig.colorbar(sc, ax=ax, label="cost", shrink=0.7)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_zlabel(f"{method.upper()}-3")
    ax.set_title(f"{type.capitalize()} ({Z.shape[-1]}D) -> 3D via {method.upper()} (color=cost)")
    plt.tight_layout()

    fig = plt.gcf()

    if show:
        plt.show()
    if save_path is not None:
        return fig
    

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def _fig_to_pil(fig):
    # Figure -> RGB 이미지로 렌더링
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(buf)

def _hstack_images(imgs):
    # PIL 이미지들을 한 행으로 이어붙이기
    heights = [im.height for im in imgs]
    max_h = max(heights)
    total_w = sum(im.width for im in imgs)

    out = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    return out


def plot_latent_cost_geometry(Z: np.ndarray, cost: np.ndarray, method: str = "pca", type:str = "latent", seed: int = 0, save_path: str = None, show: bool = True, **embed_kwargs):
    """
    Convenience function to plot all three types of latent-cost graphs.
    """
    fig1 = plot_2d(Z, cost, method=method, type=type, seed=seed, save_path=save_path, show=show, **embed_kwargs)
    fig2 = plot_3d_height(Z, cost, method=method, seed=seed, save_path=save_path, show=show, **embed_kwargs)
    fig3 = plot_3d_embed(Z, cost, method=method, type=type, seed=seed, save_path=save_path, show=show, **embed_kwargs)

    if save_path is not None:
        img = _hstack_images([_fig_to_pil(fig1), _fig_to_pil(fig2), _fig_to_pil(fig3)])
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        img.save(save_path)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Z: (N,64), cost: (N,)
    # Z, cost를 네 데이터로 바꿔서 실행
    N = 200
    Z = np.random.randn(N, 64)
    cost = (Z[:, 0] * 0.7 - Z[:, 1] * 0.3) + 0.2*np.random.randn(N)

    plot_2d(Z, cost, method="pca")
    plot_3d_height(Z, cost, method="pca")
    plot_3d_embed(Z, cost, method="pca")

    # t-SNE 예시 (느림, N이 크면 빡셈)
    # plot_2d(Z, cost, method="tsne", perplexity=30)
