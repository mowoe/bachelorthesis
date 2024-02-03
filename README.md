# Bachelor Thesis

This repository holds the accompanying code for my bachelor thesis.

The general idea is to use StableDiffusion Img2Img Metric based optimizations of screenshots of User Interfaces.

To do this, a stable diffusion model has been finetuned on a

### What has been done up to now

- Created a Stable Diffusion finetuning Dataset, which has been created with screenshots from the RICO dataset by the InteractionMining research group ([link](http://www.interactionmining.org/rico.html)), which then have been automatically annotated using the [Salesforces BLIP model](https://github.com/salesforce/BLIP). The dataset can be found on huggingface: [mowoe/rico-captions-blip-large-conditioned](https://huggingface.co/datasets/mowoe/rico-captions-blip-large-conditioned)
- Ported the code from the 2022 Paper "Modeling how different user groups perceive webpage aesthetics" (Luis A. Leiva, Morteza Shiripour, Antti Oulasvirta [doi:10.1007/s10209-022-00910-x](https://link.springer.com/article/10.1007/s10209-022-00910-x)) to use Pytorch while preserving the proposed model architecture. Model has been trained again and published on huggingface too: [mowoe/modeling_how_different_user_groups_model](https://huggingface.co/mowoe/modeling_how_different_user_groups_model)
- Finetuned a Stable Diffusion model to use the mentioned dataset, huggingface: [mowoe/stable-diffusion-v1-4-rico-blip-large-conditioned](https://huggingface.co/mowoe/stable-diffusion-v1-4-rico-blip-large-conditioned)

### Outlook
- Use [WebSight](https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned) (img2html) models to generate valid code from optimized screenshot
- Introduce penalty when semantic meaning changes too much ()
- Use some text detection model to fix malformed text in generated screenshots

## Repo Structure
