from myInference import generate_audio, save_audio, free_memory
from stable_audio_tools.models.localPretrained import get_pretrained_model_local

model, model_config = get_pretrained_model_local("./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json", "C:/Users/gabri/Downloads/models/dreamt_14.ckpt")

print("Model loaded")

audio, sr = generate_audio(
    model,
    prompt="Smooth and seductive at 115 BPM trap beat with electric guitar riffs, plucked bass, vocal adlibs, and warm synth pads. Relaxed, romantic, and sexy mood.",
    cfg_scale=7.0,
    steps=200,
    start_sec=0,
    duration_sec=47.0,
    seed=1406776750
)

print("End of generation. Saving audio...")

save_audio("output.wav", audio, sr)
free_memory()
