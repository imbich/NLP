import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_dimensions(vectors: np.ndarray, method: str = None, n_components: int = 2, **kwargs):
    """Giảm chiều bằng PCA hoặc t-SNE. Trả về ma trận (n_samples, n_components)."""
    if vectors.size == 0:
        return np.zeros((0, n_components))
    if method is None:
        pca = PCA(n_components=n_components, random_state=0)
        tsne = TSNE(n_components=n_components, random_state=0, **kwargs)
        return pca.fit_transform(vectors), tsne.fit_transform(vectors)
    
    if method.lower() == "pca":
        pca = PCA(n_components=n_components, random_state=0)
        return pca.fit_transform(vectors)
    elif method.lower() == "tsne":
        tsne = TSNE(n_components=n_components, random_state=0, **kwargs)
        return tsne.fit_transform(vectors)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

def plot_embeddings(words, pca_points, tsne_points, dim=2, title_prefix="Embeddings", figsize=(12, 5)):
    """Vẽ PCA và t-SNE song song (2D hoặc 3D)."""
    if dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        titles = [f"{title_prefix} - PCA 2D", f"{title_prefix} - t-SNE 2D"]

        for ax, data, t in zip(axes, [pca_points, tsne_points], titles):
            x, y = data[:, 0], data[:, 1]
            ax.scatter(x, y, s=30, c="lightgreen")
            for i, word in enumerate(words[:50]):  # tránh quá nhiều nhãn
                ax.text(x[i]+0.02, y[i]+0.02, word, fontsize=8)
            ax.set_title(t)
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    elif dim == 3:
        fig = plt.figure(figsize=figsize)
        titles = [f"{title_prefix} - PCA 3D", f"{title_prefix} - t-SNE 3D"]

        for i, data in enumerate([pca_points, tsne_points], 1):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            ax.scatter(x, y, z, s=30, c="lightblue")
            for j, word in enumerate(words[:50]):
                ax.text(x[j], y[j], z[j], word, fontsize=7)
            ax.set_title(titles[i-1])
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("dim phải là 2 hoặc 3")
    
def plot_2d(words, points, title="Embeddings 2D", figsize=(8, 6)):
    """Vẽ scatter 2D và chú thích từ."""
    plt.figure(figsize=figsize)
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, s=30, c="lightgreen")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_3d(words, points, title="Embeddings 3D", figsize=(10, 8)):
    """Vẽ scatter 3D và chú thích từ."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, s=30, c="lightblue")
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":

    path = r"C:\Users\NGUYEN PHUONG BICH\HOC_TAP\NLP\Data\glove.2024.wikigiga.50d\wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

    word_vec = dict()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:500]
        for line in lines:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            word_vec[word] = vector
    
    vectors = np.vstack(list(word_vec.values()))
    words = list(word_vec.keys())
        # giảm chiều và vẽ 2D + 3D
    pca_2d, tsne_2d = reduce_dimensions(vectors, n_components=2, perplexity=30)
    plot_embeddings(words, pca_2d, tsne_2d, dim=2, title_prefix="Word Embeddings")

    pca_3d, tsne_3d = reduce_dimensions(vectors, n_components=3, perplexity=30)
    plot_embeddings(words, pca_3d, tsne_3d, dim=3, title_prefix="Word Embeddings")
    # reduced_2d = reduce_dimensions(vectors, method="pca", n_components=2)
    # reduced_3d = reduce_dimensions(vectors, method="pca", n_components=3)

    # plot_2d(list(word_vec.keys()), reduced_2d)
    # plot_3d(list(word_vec.keys()), reduced_3d)