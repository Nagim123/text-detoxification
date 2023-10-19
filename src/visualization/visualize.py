import matplotlib.pyplot as plt
import json
import argparse
import pathlib
import os
import numpy as np

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
FIGURES_PATH = os.path.join(SCRIPT_PATH, "../../reports/figures")

def visualize_metrics(metric_data: dict, model_name: str) -> None:
    """
    Visualize metrics of a model using boxplot.

    Parameters:
        metric_data (dict): Dictionary with model's evaluation metrics.
        model_name (str): Name of model that generate this data.
    """
    
    # Create lists of metrics
    bleu_scores = []
    ter_scores = []
    rouge1_f1_scores = []
    rouge2_f1_scores = []

    # Collect metrics from metric data
    for entry in metric_data:
        bleu_scores.append(entry["BLEU"])
        ter_scores.append(entry["TER score"])
        rouge1_f1_scores.append(entry["ROUGES"]["rouge1_fmeasure"])
        rouge2_f1_scores.append(entry["ROUGES"]["rouge2_fmeasure"])

    # Create figure
    fig = plt.figure(figsize=(5, 7)) #(1, 4, figsize=(9, 4))
    gs = fig.add_gridspec(2, 3)
    
    bins = np.arange(0, 1, 0.1)
    # Draw boxplots
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(bleu_scores, bins=bins, edgecolor="w")
    ax1.set_title("BLEU")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.boxplot(ter_scores)
    ax2.set_title("TER")
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot(rouge1_f1_scores)
    ax3.set_title("rouge1")
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.boxplot(rouge2_f1_scores)
    ax4.set_title("rouge2")
    
    # Set title and padding
    fig.tight_layout(pad=3.0)
    fig.suptitle(f"Metrics for {model_name}")
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_PATH, f"{model_name}_metrics.png"))

def visaulize_loss_plots(model_name: str) -> None:
    """
    Visualize average loss during epochs using simple plot.

    Parameters:
        model_name (str): Name of model that generate this data.
    """
    # Average losses per epoch obtained from Kaggle notebook output during training.
    models_losses = {
        "lstm": {
            "train_loss": [4.454, 4.081, 3.913, 3.786, 3.680, 3.582, 3.493, 3.412, 3.340, 3.274],
            "val_loss": [4.167, 4.010, 3.943, 3.924, 3.933, 3.963, 4.005, 4.046, 4.101, 4.162]
        },
        "ae_lstm": {
            "train_loss": [1,2,3,4],
            "val_loss": [1,3,3,4]
        },
        "transformer": {
            "train_loss": [3.172, 2.481, 2.290, 2.174, 2.089, 2.021, 1.966, 1.918, 1.876, 1.839, 1.805, 1.775, 1.746, 1.720, 1.696],
            "val_loss": [2.560, 2.358, 2.258, 2.195, 2.159, 2.124, 2.106, 2.086, 2.073, 2.064, 2.060, 2.051, 2.048, 2.048, 2.049],
        }
    }

    # Create new figure
    plt.figure(2)
    # List of epochs numbers
    epochs = [i for i in range(len(models_losses[model_name]["train_loss"]))]
    # Plot loss/epoch plot
    plt.plot(epochs, models_losses[model_name]["train_loss"], label="Train loss")
    plt.plot(epochs, models_losses[model_name]["val_loss"], label="Validation loss")
    # Set labels and titles
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"{model_name} losses")
    plt.legend()
    # Save figure
    plt.savefig(os.path.join(FIGURES_PATH, f"{model_name}_losses.png"))


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Visualize data about model's performance and training")
    parser.add_argument("--metric_file", type=str)
    parser.add_argument("--plot_losses", action="store_true")
    parser.add_argument("--model_name", type=str, default="unknown")
    args = parser.parse_args()

    # If metric file provided
    if not args.metric_file is None:
        # Read it and then visualize the data from it
        with open(args.metric_file, "r") as metric_file:
            metric_data = json.loads(metric_file.read())
            visualize_metrics(metric_data, args.model_name)
    
    # If user asks to plot losses then plot them
    if args.plot_losses:
        visaulize_loss_plots(args.model_name)