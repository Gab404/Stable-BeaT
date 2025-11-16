import json
import os


from torch import float16
from .factory import create_model_from_config
from .utils import load_ckpt_state_dict


def get_pretrained_model_local(config_path: str, ckpt_path: str):
    """
    Charge un modèle Stable Audio depuis :
      - un fichier JSON local de configuration
      - un fichier checkpoint local (.safetensors ou .ckpt)

    Args:
        config_path (str): chemin vers model_config.json
        ckpt_path (str): chemin vers model.safetensors ou model.ckpt

    Returns:
        model: modèle PyTorch chargé
        model_config (dict): configuration JSON chargée
    """

    # Vérification existence
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    # Charger config
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Construire modèle
    model = create_model_from_config(model_config)

    # Charger le checkpoint
    state = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(state)

    # Mode eval + supprimer création graphes de calcul
    model = model.to("cuda").eval().requires_grad_(False)
    model.to(float16) # Quantization

    return model, model_config
