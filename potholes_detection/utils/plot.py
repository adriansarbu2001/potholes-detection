import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def plot_from_generator(generator):
    for x in generator.as_numpy_iterator():
        img = x[0][0]
        mask = x[1][0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

        ax1.imshow(img)
        ax1.contour(mask.squeeze(), colors="k", linewidths=5, levels=[0.5])
        ax1.set_title("RGB")

        ax2.imshow(mask.squeeze(), cmap="gray", interpolation="bilinear")
        ax2.set_title("Label")

        plt.show()


def plot_learning_curve(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
