import comet_ml
import torch
import tqdm
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps
from accelerate import Accelerator
from loguru import logger
from huggingface_hub import hf_hub_download
from transformers import BlipProcessor, BlipForConditionalGeneration

accelerator = Accelerator()
device = accelerator.device

# device = "cpu"

# IMAGE_SIZE of the dataset is roughly ~1024 by ~768, divided by two:
IMAGE_SIZE = (512, 384)

STABLE_DIFFUSION_MODEL = "mowoe/stable-diffusion-v1-4-rico-blip-large-conditioned"


def get_initial_prompt_embeddings(prompt: str, pipe: StableDiffusionImg2ImgPipeline):
    pipe._cross_attention_kwargs = None
    pipe._guidance_scale = 7.5
    pipe._clip_skip = None
    text_encoder_lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
    )
    pipe.to(device)

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        1,
        pipe.do_classifier_free_guidance,
        None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=text_encoder_lora_scale,
        clip_skip=pipe.clip_skip,
    )
    return prompt_embeds, negative_prompt_embeds


def preprocess_image(path):
    transform = Compose([
        Resize(IMAGE_SIZE),
        ToTensor(),
    ])
    image = Image.open(path).convert('RGB')
    start_image = transform(image)
    return start_image


def create_tensor_image_from_diffusion_pipeline(start_image: torch.Tensor, prompt_embeds, negative_prompt_embeds,
                                                pipe: StableDiffusionImg2ImgPipeline, num_inference_steps=40):
    """
    Broken down from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
    in order to not convert the tensor image into a pillow image,
    to keep autograd functionality
    :param start_image:
    :param prompt_embeds:
    :param negative_prompt_embeds:
    :param pipe:
    """
    batch_size = 1
    num_images_per_prompt = 1
    strength = 0.55
    output_type = "pil"
    generator = None
    timesteps = None
    eta = 0.0
    guidance_scale = 7.5
    pipe._guidance_scale = guidance_scale
    pipe._interrupt = False
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Preprocess image
    image = pipe.image_processor.preprocess(start_image)

    # 5. set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps)
    timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 6. Prepare latent variables
    latents = pipe.prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Add image embeds for IP-Adapter
    added_cond_kwargs = None

    # 7.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    tensor_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    logger.info("Image shape: {}".format(image.shape))
    logger.info("Image dtype: {}".format(image.dtype))

    image = pipe.image_processor.postprocess(tensor_image.detach(), output_type=output_type,
                                             do_denormalize=[True] * image.shape[0])[0]
    logger.info("PIL Image shape: {}".format(image.size))

    # Offload all models
    pipe.maybe_free_model_hooks()
    return tensor_image


class PromptModel(torch.nn.Module):
    def __init__(self, start_image: str):
        super().__init__()
        self.transform = Compose([
            Resize(IMAGE_SIZE),
            ToTensor(),
        ])
        image = Image.open(start_image).convert('RGB')
        self.start_image = self.transform(image)

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16)

        initial_caption = self.get_blip_annotation(start_image)
        initial_prompt, initial_negative_prompt = get_initial_prompt_embeddings(initial_caption, self.pipe)
        self.fc1 = torch.nn.Parameter(initial_prompt)
        self.fc2 = torch.nn.Parameter(initial_negative_prompt)

    def get_blip_annotation(self, path: str):
        raw_image = Image.open(path).convert('RGB')
        text = "a mobile screen showing"
        inputs = self.blip_processor(raw_image, text, return_tensors="pt").to(device, torch.float16)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"Blip caption for {path}: {caption}")
        return caption

    def forward(self):
        image = create_tensor_image_from_diffusion_pipeline(
            self.start_image,
            self.fc1,
            self.fc2,
            self.pipe
        )
        return image


class AestheticPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(32, 4096)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(4096, 1)
        # Download the model from HF Hub
        local_filename = hf_hub_download(repo_id="mowoe/modeling_how_different_user_groups_model", filename="model.pt")
        self.load_state_dict(torch.load(local_filename, map_location="cpu"))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Use BLIP interrogation for the initial starting image to form a starting prompt
# Optimize from there
# Calculate the aesthetic score for the generated image
# Apply penalty with classifier (prevent generation from generating something other than a UI): TODO

if __name__ == "__main__":
    experiment = comet_ml.Experiment(
        api_key="<Your API Key>",
        project_name="mowoe/bachelor-thesis"
    )
    prompt_model = PromptModel("./0.png").to(device)
    aesthetic_predictor = AestheticPredictor().to(device, torch.float16)
    optimizer = torch.optim.SGD([prompt_model.fc1], lr=0.005, momentum=0.9)
    for x in tqdm.tqdm(range(50)):
        optimizer.zero_grad()
        generated_image = prompt_model()
        score = aesthetic_predictor(generated_image)
        logger.info(f"Score: {score}")
        loss = -score
        loss.backward()
        optimizer.step()
        tensor_img = create_tensor_image_from_diffusion_pipeline(
            prompt_model.start_image,
            prompt_model.fc1.detach().to(device, torch.float16),
            prompt_model.fc2.detach().to(device, torch.float16),
            prompt_model.pipe
        ).detach()
        pil_image = prompt_model.pipe.image_processor.postprocess(tensor_img, output_type="pil",
                                                                  do_denormalize=[True] * tensor_img.shape[0])[0]
        pil_image.save(f"./output_{x}.png")
        experiment.log_image(f"./output_{x}.png", step=x)
        experiment.log_metric("aesthetic_score", score[0].item, step=x)

    tensor_img = create_tensor_image_from_diffusion_pipeline(
        prompt_model.start_image,
        prompt_model.fc1.detach().to(device, torch.float16),
        prompt_model.fc2.detach().to(device, torch.float16),
        prompt_model.pipe,
        num_inference_steps=50
    ).detach()
    pil_image = prompt_model.pipe.image_processor.postprocess(tensor_img, output_type="pil", do_denormalize=[True] * tensor_img.shape[0])[0]
    pil_image.save("./output.png")
    experiment.end()
