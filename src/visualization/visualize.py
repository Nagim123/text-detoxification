import matplotlib.pyplot as plt
import json
import argparse

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

    # Create 4 subplots
    figure, axis = plt.subplots(1, 4, figsize=(9, 4))

    # Draw boxplots
    axis[0].boxplot(bleu_scores)
    axis[0].set_title("BLEU")
    axis[1].boxplot(ter_scores)
    axis[1].set_title("TER")
    axis[2].boxplot(rouge1_f1_scores)
    axis[2].set_title("rouge1")
    axis[3].boxplot(rouge2_f1_scores)
    axis[3].set_title("rouge2")
    
    # Set title and padding
    figure.tight_layout(pad=3.0)
    figure.suptitle(f"Metrics for {model_name}")
    
    # Save figure
    plt.savefig(f"{model_name}_metrics.png")

def visaulize_loss_plots(model_name: str) -> None:
    """
    Visualize average loss during epochs using simple plot.

    Parameters:
        model_name (str): Name of model that generate this data.
    """
    # Average losses per epoch obtained from Kaggle notebook output during training.
    models_losses = {
        "transformer": [3.0, 2.5, 2.3, 2.2, 2.1, 1.9, 1.85, 1.7]
    }

    # Create new figure
    plt.figure(2)
    # List of epochs numbers
    epochs = [i for i in range(len(models_losses[model_name]))]
    # Plot loss/epoch plot
    plt.plot(epochs, models_losses[model_name])
    # Set labels and titles
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"{model_name} losses")
    # Save figure
    plt.savefig(f"{model_name}_losses.png")


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