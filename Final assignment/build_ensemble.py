import torch
from model_ensemble import EnsembleModel

if __name__ == "__main__":
    # Path to your best checkpoint from "train.py"
    best_checkpoint_path = "checkpoints/best_models_epoch=XX_val=YY.pth"

    # Load the checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")

    # Build the ensemble
    ensemble = EnsembleModel()

    # Load the submodels' weights from the checkpoint
    ensemble.model_small.load_state_dict(checkpoint["model_small"])
    ensemble.model_medium.load_state_dict(checkpoint["model_medium"])
    ensemble.model_big.load_state_dict(checkpoint["model_big"])

    # Now save the entire ensemble as a single .pth
    # This is a single file you can upload to CodaLab.
    torch.save(ensemble.state_dict(), "ensemble_model.pth")
