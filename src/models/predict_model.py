import torch
import argparse
import os
import json

from core.utils.text_manager import TextManager
from core.utils.custom_predict import predict_using_custom_models
from core.utils.huggingface_predict import predict_using_external_models
from core.utils.constants import MODEL_WEIGHTS_PATH
from core.utils.metrics_calculator import generate_metric_evalulation
from core.architectures import lstm, ae_lstm, transformer, t5_paranmt


if __name__ == "__main__":
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of available models
    available_models = {
        "lstm": lstm.DetoxificationModel(),
        "ae_lstm": ae_lstm.DetoxificationModel(device),
        "transformer": transformer.DetoxificationModel(device),
        "T5": t5_paranmt.DetoxificationModel(),
    }
    
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Do prediction for any model that are supported")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("weights", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("--compare", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    # Create preprocessor with dataset vocabulary
    toxic_text_manager = TextManager(args.file_path, args.compare, device)

    # Load model 
    model = available_models[args.model_type]
    weights_path = os.path.join(MODEL_WEIGHTS_PATH, args.weights)

    # Load weigths and make predictions
    if args.model_type == "T5":
        model.load_model()    
        results = predict_using_external_models(model, toxic_text_manager)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        results = predict_using_custom_models(model, toxic_text_manager, device)
    
    if not args.out_dir is None:
        if args.compare is None:
            with open(args.out_dir, "w") as output_file:
                output_file.write("\n".join(results))
        else:
            evaluation_results = generate_metric_evalulation(results, toxic_text_manager)
            with open(args.out_dir, "w") as output_file:
                output_file.write(json.dumps(evaluation_results))
    else:
        print(results)