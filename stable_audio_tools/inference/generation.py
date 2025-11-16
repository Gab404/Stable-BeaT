import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop

def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    # Now the generative AI part:

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, device=device)

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    """
    Generate audio from a prompt using a diffusion model.

    Supports optional audio conditioning via `conditioning["audio"] = (sr, tensor)`
    and optional init_audio for variation/inpainting via `init_audio = (sr, tensor)`.

    Args:
        model: diffusion model
        steps: number of diffusion steps
        cfg_scale: classifier free guidance scale
        conditioning: dict of conditioning values; may include "audio": (sr, tensor)
        conditioning_tensors: precomputed conditioning tensors (optional)
        negative_conditioning: dict for negative prompts (optional)
        negative_conditioning_tensors: precomputed neg cond tensors (optional)
        batch_size, sample_size, sample_rate, seed, device: standard
        init_audio: (sr, tensor) used as initial variation seed / inpainting
        init_noise_level: noise strength when using init_audio
        return_latents: return latents instead of decoded audio
        **sampler_kwargs: forwarded to sampler
    """
    # length of the output in audio samples (decoded audio)
    audio_sample_size = sample_size

    # If latent diffusion, convert sample_size to latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    # Seed
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)

    # initial noise
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    # disable some TF32 behaviors for determinism / stability (kept from original)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning: either tensors provided or compute via model.conditioner()
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"

    # If user gave conditioning dict and not precomputed tensors -> compute
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    else:
        # ensure it's a dict copy we can modify safely
        conditioning_tensors = dict(conditioning_tensors)

    # --- AUDIO CONDITIONING: if a tuple (sr, tensor) is provided under conditioning["audio"],
    # encode / prepare it and insert into conditioning_tensors under a chosen id "audio_embedding" ---
    # NOTE: for this to actually affect generation, the model must have been trained to
    # accept "audio_embedding" as a conditioning id (e.g. included in cross_attention_cond_ids).
    if conditioning is not None and "audio" in conditioning and conditioning["audio"] is not None:
        try:
            audio_sr, audio_tensor = conditioning["audio"]
            # prepare_audio will resample, pad/crop and set channels
            # target length must be the decoded audio length (audio_sample_size),
            # but if model.pretransform exists we want the decoded shape prior to encoding.
            target_length = audio_sample_size
            target_channels = model.io_channels
            # When encoding later, the pretransform.encode expects decoded audio shape.
            prepared_audio = prepare_audio(audio_tensor, in_sr=audio_sr,
                                           target_sr=model.sample_rate,
                                           target_length=target_length,
                                           target_channels=target_channels,
                                           device=device)
            # If model expects latent conditioning (e.g. audio in latent space), encode
            if model.pretransform is not None:
                # encode to latent space used by model
                audio_latent = model.pretransform.encode(prepared_audio)
                # audio_latent shape: [C_latent, L_latent] -> add batch dim
                audio_latent = audio_latent.repeat(batch_size, 1, 1)
                conditioning_tensors["audio_embedding"] = [audio_latent]
            else:
                # keep waveform (batch, channels, length)
                prepared_audio = prepared_audio.repeat(batch_size, 1, 1)
                conditioning_tensors["audio_embedding"] = [prepared_audio]
        except Exception as e:
            # if something goes wrong, raise a clear error
            raise RuntimeError(f"Error preparing audio conditioning: {e}")

    # Negative conditioning (precompute tensors if needed)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
        # ensure copy
        negative_conditioning_tensors = dict(negative_conditioning_tensors)
        # If user provided negative conditioning audio, process it the same way as above
        if negative_conditioning is not None and "audio" in negative_conditioning and negative_conditioning["audio"] is not None:
            try:
                naudio_sr, naudio_tensor = negative_conditioning["audio"]
                n_prepared = prepare_audio(naudio_tensor, in_sr=naudio_sr,
                                          target_sr=model.sample_rate,
                                          target_length=audio_sample_size,
                                          target_channels=model.io_channels,
                                          device=device)
                if model.pretransform is not None:
                    n_audio_latent = model.pretransform.encode(n_prepared)
                    n_audio_latent = n_audio_latent.repeat(batch_size, 1, 1)
                    negative_conditioning_tensors["audio_embedding"] = [n_audio_latent]
                else:
                    n_prepared = n_prepared.repeat(batch_size, 1, 1)
                    negative_conditioning_tensors["audio_embedding"] = [n_prepared]
            except Exception as e:
                raise RuntimeError(f"Error preparing negative audio conditioning: {e}")
    else:
        negative_conditioning_tensors = {}

    # If user passed init_audio (variation / inpainting), prepare and optionally encode it
    if init_audio is not None:
        in_sr, init_audio_tensor = init_audio

        io_channels = model.io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        init_audio_prepared = prepare_audio(init_audio_tensor,
                                           in_sr=in_sr,
                                           target_sr=model.sample_rate,
                                           target_length=audio_sample_size,
                                           target_channels=io_channels,
                                           device=device)
        # Encode to latents if model uses pretransform
        if model.pretransform is not None:
            init_audio_latent = model.pretransform.encode(init_audio_prepared)
            init_audio = init_audio_latent.repeat(batch_size, 1, 1)
        else:
            init_audio = init_audio_prepared.repeat(batch_size, 1, 1)

        # use init_noise_level to limit sigma
        sampler_kwargs["sigma_max"] = init_noise_level
    else:
        init_audio = None

    # convert conditioning_tensors into conditioning_inputs for the sampler
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    if negative_conditioning_tensors:
        negative_conditioning_inputs = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_inputs = {}

    # cast everything to model dtype
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    # ensure conditioning_inputs tensors are same dtype or None
    conditioning_inputs = {k: (v.type(model_dtype) if v is not None else v) for k, v in conditioning_inputs.items()}
    negative_conditioning_inputs = {k: (v.type(model_dtype) if v is not None else v) for k, v in negative_conditioning_inputs.items()}

    # call the sampler (k-diffusion or rectified flow)
    diff_objective = model.diffusion_objective

    if diff_objective == "v":
        sampled = sample_k(
            model.model,
            noise,
            init_audio,
            steps,
            **sampler_kwargs,
            **conditioning_inputs,
            **negative_conditioning_inputs,
            cfg_scale=cfg_scale,
            batch_cfg=True,
            rescale_cfg=True,
            device=device
        )
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]
        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(
            model.model,
            noise,
            init_data=init_audio,
            steps=steps,
            **sampler_kwargs,
            **conditioning_inputs,
            **negative_conditioning_inputs,
            dist_shift=getattr(model, "dist_shift", None),
            cfg_scale=cfg_scale,
            batch_cfg=True,
            rescale_cfg=True,
            device=device
        )
    else:
        raise RuntimeError(f"Unsupported diffusion objective: {diff_objective}")

    # cleanup
    del noise
    # remove big refs to conditioning_tensors -> allow GC
    try:
        del conditioning_tensors
    except Exception:
        pass
    try:
        del conditioning_inputs
    except Exception:
        pass
    torch.cuda.empty_cache()

    # decode latents back to audio if needed
    if model.pretransform is not None and not return_latents:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    return sampled


def generate_diffusion_cond_inpaint(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion inpainting model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A mask to use for inpainting. Shape should be [batch_size, sample_size]
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask