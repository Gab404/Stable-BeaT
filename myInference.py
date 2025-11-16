import torch
import numpy as np
from einops import rearrange
import torchaudio

from stable_audio_tools.inference.generation import generate_diffusion_cond


# -------------------------------------------------------------
# Fonction d’inférence simple
# -------------------------------------------------------------
def generate_audio(
    model,
    prompt: str,
    cfg_scale: float = 7.0,
    steps: int = 100,
    start_sec: float = 0.0,
    duration_sec: float = 47.0,
    seed: int = -1,
    sampler_type: str = "dpmpp-3m-sde",
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_rate: int = 44100,
    sample_size: int = 2097152,
    device: str = "cuda",
):
    device = torch.device(device)

    # Déterminer seed
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)

    # Construire le conditioning
    conditioning_dict = {
        "prompt": prompt,
        "seconds_start": float(start_sec),
        "seconds_total": float(duration_sec),
    }
    conditioning = [conditioning_dict]

    # Aucun negative prompt ici
    negative_conditioning = None

    # Appeler la génération interne
    with torch.no_grad():
        audio = generate_diffusion_cond(
            model=model,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            cfg_interval=(0.0, 1.0),
            batch_size=1,
            sample_size=sample_size,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=None,
            init_noise_level=1.0,
            callback=None,
            scale_phi=cfg_rescale,
            rho=rho,
        )

    # audio : tensor [1, 2, N] ou [1, 1, N]

    # normalisation / clamp comme Gradio
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32)
    audio = audio / torch.max(torch.abs(audio))
    audio = audio.clamp(-1, 1)

    # conversion numpy
    audio = audio.cpu().numpy()

    return audio, sample_rate


# -------------------------------------------------------------
# Sauvegarde simple
# -------------------------------------------------------------
def save_audio(path, audio, sample_rate):
    tensor = torch.from_numpy(audio)
    torchaudio.save(path, tensor, sample_rate)

def free_memory():
    torch.cuda.empty_cache()
