import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
import pandas as pd

plt.style.use("ggplot")
save_path = "C:/Users/Yannick/OneDrive/Dokumente/Studium/Masterarbeit/Ausarbeitung/98_images/3d_params_new.pgf"
input_file_path = "C:/Users/Yannick/OneDrive/Dokumente/Studium/Masterarbeit/Ausarbeitung/98_images/source/hparams_table_counting_full_cold_new.csv"


def main():
    df = pd.read_csv(input_file_path)
    print(df.head())

    print(df.loc[df["number_params"] == df["number_params"].max()])
    df.drop(axis=1, index=52, inplace=True)
    x = df["units"].values
    y = df["kernel_number"].values
    z = df["kernel_size"].values
    color = df["number_params"].values

    visualize_input_3d(x, y, z, color)


def visualize_input_3d(x, y, z, c):
    """Visualize input in 3D (x, y, z)"""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    im = ax.scatter(xs=x, ys=y, zs=z, c=c, alpha=1.0, cmap=plt.get_cmap("inferno"))
    ax.set_xlabel("Einheiten innerhalb einer RNN-Zelle")
    ax.set_ylabel("Anzahl Filterkerne pro Layer")
    ax.set_zlabel("Größe der Filterkerne")
    ax.view_init(elev=10.0, azim=45.0)

    fig.colorbar(im, ax=ax, label="Anzahl der Parameter des Modells")
    plt.show()
    matplotlib.rcParams.update(
        {"pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True, "pgf.rcfonts": False}
    )
    fig.savefig(save_path)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


main()
