import numpy as np
import matplotlib.pyplot as plt

def plot_paths(t, S, title="Trajectoires"):
    plt.figure(figsize=(10, 6))
    for i in range(min(20, S.shape[0])):
        plt.plot(t, S[i])
        plt.grid(True)
        plt.title(title)
        plt.xlabel("Temps")
        plt.ylabel("Prix")
        plt.show()

def plot_hist(x, bins=60, title="Histogramme"):
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, density=True)
    plt.grid(True)
    plt.title(title)
    plt.show()
