import json
from math import prod
from os import listdir
from os.path import isfile, join

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


from utils import (
    AestheticPredictor,
    calculate_initial_theta,
    get_all_bounding_boxes,
    get_random_initial_position,
    segment,
    stack_alpha_aware,
)

# RICO_COMBINED_LOCATION = "/Volumes/data/datasets/combined"
# RICO_COMBINED_LOCATION = "/mnt/ceph/storage/data-tmp/current/sz46wone/combined"
RICO_COMBINED_LOCATION = "/var/tmp/sz46wone/combined/combined"
#OUTPUT_DIR = "/Volumes/data/datasets/real_and_fake_rico_layouts/train"
OUTPUT_DIR = "/var/tmp/sz46wone/real_and_fake_rico_layouts/train"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(join(OUTPUT_DIR,"real")).mkdir(parents=True, exist_ok=True)
Path(join(OUTPUT_DIR,"fake")).mkdir(parents=True, exist_ok=True)


transform_t_to_pil = T.ToPILImage()
transform_to_t = transforms.Compose([transforms.ToTensor()])

onlyfiles = [
    f
    for f in listdir(RICO_COMBINED_LOCATION)
    if isfile(join(RICO_COMBINED_LOCATION, f)) and ("jpg" in f or "jpeg" in f)
]


def get_image_from_coordinates(
    coordinates,
    segments_and_positions,
    original_image_size,
    background_color=torch.Tensor([1.0, 1.0, 1.0]),
) -> Image:
    segments_on_canvas = []
    canvas_size = (1, 3, original_image_size[1], original_image_size[0])

    # Create background image from parameter
    bg_col = torch.clamp(background_color, min=0, max=1)
    # print(f"Clamped bg to {bg_col}")
    red = torch.tile(bg_col[0], original_image_size[::-1])
    green = torch.tile(bg_col[1], original_image_size[::-1])
    blue = torch.tile(bg_col[2], original_image_size[::-1])
    alpha = torch.tile(torch.tensor(0.0), original_image_size[::-1])
    background = torch.stack([red, green, blue, alpha]).unsqueeze(0)
    # print(background.detach().sum())
    # background = torch.tile(background_color,original_image_size)

    for n in range(len(segments_and_positions)):

        # We need to calculate the proper ratios, to artificially warp the segment on to a bigger canvas without distorting it (see notebook 01)
        x_ratio = original_image_size[0] / segments_and_positions[n][0].size[0]
        y_ratio = original_image_size[1] / segments_and_positions[n][0].size[1]

        # Affine matrix
        theta = [[x_ratio, 0.0, 0.0], [0.0, y_ratio, 0.0]]
        theta_tensor = torch.as_tensor(theta)[None]

        theta_tensor[0][0][2] += coordinates[n][0]
        theta_tensor[0][1][2] += coordinates[n][1]

        # Generate flow field
        grid = F.affine_grid(theta_tensor, canvas_size, align_corners=False).type(torch.FloatTensor)
        x = F.grid_sample(
            transform_to_t(segments_and_positions[n][0]).unsqueeze(0), grid, align_corners=False
        )
        segments_on_canvas.append(x)

    segments_on_canvas.append(background)

    generated_image = stack_alpha_aware(segments_on_canvas)
    generated_image = generated_image[:3]

    img = transform_t_to_pil(generated_image)
    return img

def process_file(file):
    logger.info(f"Processing {file}...")
    im = Image.open(join(RICO_COMBINED_LOCATION, file))
    im = im.convert("RGBA")
    im = im.resize((1440, 2560), Image.Resampling.LANCZOS)
    try:
        with open(join(RICO_COMBINED_LOCATION, file).replace("jpg", "json"), "r") as f:
            image_json = json.load(f)
    except FileNotFoundError:
        logger.warning(f"{file.replace('jpg', 'json')} not available, probably not unpacked yet...")
        return

    reduced_segments = [
        s
        for s in segment(im, image_json)
        if (prod(s[0].size) > 1)
        #if (prod(s[0].size) < 0.80 * prod(im.size)) and (prod(s[0].size) > 1)
    ]

    initial_vector = []
    for seg, position in reduced_segments:
        initial_theta = calculate_initial_theta(seg, im.size, position)
        initial_vector.append([initial_theta[0][2], initial_theta[1][2]])
    coordinates = torch.tensor(initial_vector)

    logger.info("Reassembling original image...")
    image = get_image_from_coordinates(coordinates, reduced_segments, im.size)
    image.save(join(OUTPUT_DIR, f"real/{file.split('.')[0]}_original.jpg"))
    logger.info(f"Saved image to real/{file.split('.')[0]}_original.jpg")

    logger.info("Reassembling 4 random images...")
    for x in range(4):
        initial_vector = []
        for seg, position in reduced_segments:
            initial_theta = get_random_initial_position(seg, im.size, position)
            initial_vector.append([initial_theta[0][2], initial_theta[1][2]])
        coordinates = torch.tensor(initial_vector)
        image = get_image_from_coordinates(coordinates, reduced_segments, im.size)
        image.save(join(OUTPUT_DIR, f"fake/{file.split('.')[0]}_random_{x}.jpg"))

def run(files):
    with Pool(4) as pool:
        list(tqdm(pool.imap(process_file, files), total=len(files)))

if __name__ == "__main__":
    run(onlyfiles)
