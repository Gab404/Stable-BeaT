import argparse
from stable_audio_tools.inference.inference import generate_audio, save_audio, free_memory
from stable_audio_tools.models.localPretrained import get_pretrained_model_local

CONFIG_JSON = "./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json"

parser = argparse.ArgumentParser(description="Beat generation with SAO fine tuned")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for audio conditioning")
parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale")
parser.add_argument("--steps", type=int, default=200, help="Step number")
parser.add_argument("--model-ckpt", type=str, required=True, help="Pretrained model checkpoint")
parser.add_argument("--seed", type=int, default=-1, help="Seed")
parser.add_argument("--start-sec", type=float, default=0.0, help="Start second")
parser.add_argument("--duration", type=float, default=47.0, help="Duration")
parser.add_argument("--output", type=str, default="output.wav", help="Output path")

args = parser.parse_args()

model, model_config = get_pretrained_model_local(
    CONFIG_JSON,
    args.model_ckpt
)

print("Model loaded")

audio, sr = generate_audio(
    model,
    prompt=args.prompt,
    cfg_scale=args.cfg,
    steps=args.steps,
    start_sec=args.start_sec,
    duration_sec=args.duration,
    seed=args.seed
)

print("End of generation. Saving audio...")

save_audio(args.output, audio, sr)
free_memory()
