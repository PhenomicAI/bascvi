from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, figsize=(10, 10), output_filename="cm.png", normalize: str = "true"):
    """Plot confusion matrix using heatmap.

    Parameters
    ----------
    figsize : tuple, optional
        Size of the figure, by default (10, 10)
    output_filename : str, optional
        Path to output file., by default "cm.png"
    normalize: str
        true, pred, all, None
    """
    # Labels which will be plotted across x and y axis
    assert len(y_true) > 0, "length not greater than zero"
    labels = sorted(list(set(y_true)))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    sns.set(color_codes=True)
    plt.figure(1, figsize=figsize)

    plt.title("Confusion Matrix")

    sns.set(font_scale=1.4)
    ax = sns.heatmap(cm, annot=False, cbar_kws={"label": "Scale"}, linewidths=0.1)

    ax.set_xticks([x + 0.5 for x in range(len(labels))])
    ax.set_yticks([y + 0.5 for y in range(len(labels))])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation="horizontal")

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()
