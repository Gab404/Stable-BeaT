<h1 align="center"> SAO fine tuning for modern beat generation</h1>
<p align="center">
As a music and AI lover I wanted to dive into the music generation technologies.
</p>

<p align="center">
  <img src="./assets/preview.gif" alt="preview" width="400"/>
</p>

<p align="center">
First, I started by exploring existing models for music generation such as Suno or Stable Audio 2.0, but I couldn't find any that could generate modern trap/rap/r&b beat as well. Then I got this idea, fine tune an open source model over a good amount of trap beat. I chose Stable Audio Open 1.0, as I found it to be the most suitable open-source foundation for this kind of task.
</p>

---

# Requirements

Requires PyTorch 2.5 or later and Python 3.10

Install the requirements by running : `pip install -r requirements.txt`

---

# How to use

Download the fine tuned model on Hugging Face [**here**](https://huggingface.co/gab-gdp/StableBeaT)

To run the Gradio interface : 

`python run_gradio.py --model-config ./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json --ckpt-path model.ckpt`

To run the prediction function : `python run_inference.py`
</br>
</br>
With the following flags: 
- `--model-ckpt` : Pretrained model checkpoint.
- `--prompt` : The textual prompt for audio generation conditioning.
  </br>
- `--steps` : The number of steps.
- `--cfg` : CFG scale.
- `--start-second` : Start second.
- `--duration` : Audio duration.
- `--output` : Output path.
- `--seed` : For a specific seed.

</br>

I recommend at least 200 steps for a good generation.

---


## Sources
- [**Stable Audio Open 1.0**](https://huggingface.co/stabilityai/stable-audio-open-1.0) - Model used.
- [**LoRAW**](https://github.com/NeuralNotW0rk/LoRAW) — Pipeline implementation for stable audio open LoRA finetuning.
- [**Stable Audio Tools**](https://github.com/Stability-AI/stable-audio-tools) — Official stability.ai framework to use stable audio open.
- [**Essentia**](https://essentia.upf.edu/models.html) - Library for music features extractions.

## Contact - Gabriel Guiet-Dupré
- [**Linkedin**](https://www.linkedin.com/in/gabriel-guiet-dupre/)
- [**GitHub**](https://github.com/Gab404)