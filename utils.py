import numpy as np
import torch
from huggingface_hub import hf_hub_download
import random

# Utility functions to read dataset
def get_all_bounding_boxes_and_ids(item):
    all_boxes = []
    if "bounds" in item.keys() and "resource-id" in item.keys():
        all_boxes.append((item["bounds"],item["resource-id"]))
    if "children" in item.keys():
        for child in item["children"]:
            for box in get_all_bounding_boxes_and_ids(child):
                all_boxes.append(box)
    return all_boxes

def get_all_bounding_boxes(item):
    bboxes = get_all_bounding_boxes_and_ids(item)
    reduced_bboxes = []
    already_seen_ids = []
    for box,r_id in bboxes:
        if r_id not in already_seen_ids:
            reduced_bboxes.append(box)
            already_seen_ids.append(r_id)
    return reduced_bboxes

"""
returns a list of the segments and a list of coordinates
"""
def segment(image: np.ndarray, rico_json: dict)->[list,list]:
    boxes = get_all_bounding_boxes(rico_json["activity"]["root"])
    segments = []
    coordinates = []
    for box in boxes:
        anchor_point = (box[0],box[1])
        width = box[2]-box[0]
        height = box[3]-box[1]
        try:
            cropped_image = image.crop((box[0],box[1],box[2],box[3]))
        except ValueError:
            continue
        # Only include segments that are less than 90% the size of the original image
        if (width*height) < 0.9*(image.size[0]*image.size[1]):
            segments.append(cropped_image)
            coordinates.append((box[0],box[1]))
    return list(zip(segments,coordinates))


def calculate_initial_theta(segment, canvas_size, original_position):
    # Theta consists of 6 values, 4 of which we have to calculate.
    x_ratio = canvas_size[0] / segment.size[0]
    y_ratio = canvas_size[1] / segment.size[1]
    # grid_location_x and grid_location_y are basically percentages of height and width and not actual coordinates
    # Because we already warp the segment onto a bigger canvas, this transformation is a bit complicated
    # grid_location_x has to be in the interval [-(x_ratio-1),(x_ratio-1)], i.e. 0 means -(x_ratio - 1) and 1440 means (x_ratio - 1)
    # We can map U ~ [0, 1] to U ~ [a, b] with u -> (a - b)*u + b
    # We first map U ~ [0, max_width] to U ~ [0,1] by dividing by max_width

    eps = 0.00001 # Avoid div by zero
    
    original_x_position = (original_position[0]) / (canvas_size[0]-segment.size[0]+eps)
    mapped_x_position = (-(x_ratio - 1) - (x_ratio - 1))*original_x_position + (x_ratio-1)

    original_y_position = (original_position[1]) / (canvas_size[1]-segment.size[1]+eps)
    mapped_y_position = (-(y_ratio - 1) - (y_ratio - 1))*original_y_position + (y_ratio-1)

    return np.array([
        [x_ratio, 0.0    , mapped_x_position],
        [0.0    , y_ratio, mapped_y_position]
    ])

def blend_images(image1, image2):
    # Ensure the images are in torch tensor format and have the same shape
    if not isinstance(image1, torch.Tensor) or not isinstance(image2, torch.Tensor):
        raise ValueError("Both images must be torch tensors")
    
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same shape")
    
    # Split the RGBA channels
    r1, g1, b1, a1 = image1.unbind(dim=0)
    r2, g2, b2, a2 = image2.unbind(dim=0)

    return torch.stack([
        r1 * a1 + r2 * (1-a1),
        g1 * a1 + g2 * (1-a1),
        b1 * a1 + b2 * (1-a1),
        torch.ones_like(r1)
    ])
    
def stack_alpha_aware(images):
    if len(images[0].shape) == 4:
        running_top_image = torch.zeros_like(images[0][0])
    else:
        running_top_image = torch.zeros_like(images[0])
    for image in images:
        if len(image.shape) == 4:
            # running_top_image = stack_a_on_b(running_top_image,image[0])
            # running_top_image = alpha_composite(running_top_image, image[0])
            running_top_image = blend_images(image[0],running_top_image)
        else:
            # running_top_image = stack_a_on_b(running_top_image,image)
            # running_top_image = alpha_composite(running_top_image, image)
            running_top_image = blend_images(image,running_top_image)
        # display(resize_image(transform_t_to_pil(running_top_image)))
    return running_top_image

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
        self.eval()

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def get_random_initial_position(segment, canvas_size, original_position, seed=1, verbose=0):
    # random.seed(seed)

    
    # Theta consists of 6 values, 4 of which we have to calculate.
    x_ratio = canvas_size[0] / segment.size[0]
    y_ratio = canvas_size[1] / segment.size[1]
    
    mapped_x_position = random.uniform(-(x_ratio-1), (x_ratio-1))
    mapped_y_position = random.uniform(-(y_ratio-1), (y_ratio-1))

    if verbose:
        print(f"Original would have been: {calculate_initial_theta(segment,canvas_size,original_position)}")
        print(f"""Now is {np.array([ 
            [x_ratio, 0.0    , mapped_x_position],
            [0.0    , y_ratio, mapped_y_position]
        ])}""")

    
    return np.array([
        [x_ratio, 0.0    , mapped_x_position],
        [0.0    , y_ratio, mapped_y_position]
    ])