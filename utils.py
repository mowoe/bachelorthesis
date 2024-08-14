import numpy as np
import torch
from huggingface_hub import hf_hub_download
import random
import torchvision.transforms as T
from torchvision import datasets, transforms
from loguru import logger
from PIL import Image


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

def map_to_range(num, x_ratio, t):
    normalized_num = (num + (x_ratio - 1)) / ((2 * (x_ratio - 1))+0.00001)
    mapped_num = normalized_num * t
    return mapped_num

def get_random_initial_position(segment, canvas_size, original_position, seed=1, verbose=0, return_coords=False):
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
    if return_coords:
        original_x_coordinate = map_to_range(mapped_x_position, x_ratio, canvas_size[0])
        original_y_coordinate = map_to_range(mapped_y_position, y_ratio, canvas_size[1])
        return [original_x_coordinate,original_y_coordinate] ,np.array([
            [x_ratio, 0.0    , mapped_x_position],
            [0.0    , y_ratio, mapped_y_position]
        ])
        

        
    return np.array([
        [x_ratio, 0.0    , mapped_x_position],
        [0.0    , y_ratio, mapped_y_position]
    ])


def get_all_clickable_resources(item, should_be_clickable):
    if item is None:
        return []
    all_boxes = []
    if "bounds" in item.keys() and "resource-id" in item.keys() and "clickable" in item.keys() and item["clickable"]==should_be_clickable and item["visible-to-user"]:
        all_boxes.append((item["bounds"],item["resource-id"]))
    if "children" in item.keys():
        for child in item["children"]:
            for box in get_all_clickable_resources(child,should_be_clickable):
                all_boxes.append(box)
    return all_boxes

def get_all_bounding_boxes_with_clickable(item, should_be_clickable, exclude_navigation_bar=False):
    bboxes = get_all_clickable_resources(item,should_be_clickable)
    reduced_bboxes = []
    already_seen_ids = []
    for box,r_id in bboxes:
        if r_id not in already_seen_ids and (r_id != "android:id/navigationBarBackground" or not exclude_navigation_bar):
            reduced_bboxes.append(box)
            already_seen_ids.append(r_id)
    return reduced_bboxes

def is_overlapping(bbox1, bbox2):
    """
    This function checks if the second bounding box overlaps the first one for more than 90% of its space.
    
    Args:
      bbox1: A list of four integers representing the first bounding box (x1, y1, x2, y2).
      bbox2: A list of four integers representing the second bounding box (x1, y1, x2, y2).
    
    Returns:
      True if the second bounding box overlaps the first one for more than 90% of its space, False otherwise.
    """
    
    # Calculate the area of the intersection between the two bounding boxes
    x_overlap = max(0, min(bbox2[2], bbox1[2]) - max(bbox1[0], bbox2[0]))
    y_overlap = max(0, min(bbox2[3], bbox1[3]) - max(bbox1[1], bbox2[1]))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of the first bounding box
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    # Calculate the area of the second bounding box
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Check if the overlap is more than 90% of the second bounding box area
    return intersection_area / bbox1_area > 0.9

"""
Segments the UI into its elements, selects 2*n of them, ensuring they arent included in each other,
sorts them by size and returns everything. Raises ValueError if not enough elements can be found.
"""
def get_first_n_sorted_elements(image, image_json, n, clickable=True):
    image = image.convert('RGBA')
    image = image.resize((1440, 2560), Image.Resampling.LANCZOS)
    
    clickable_segments = get_all_bounding_boxes_with_clickable(image_json["activity"]["root"], clickable)
    
    # Filter out all 0x0 segments
    clickable_segments = [box for box in clickable_segments if ((box[2]-box[0])*(box[3]-box[1])) > 0]

    # Sort by size ascending
    sorted_clickable_segments = sorted(clickable_segments, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))

    # Filter all elements out that 'cover' at least 80% of some other element (the smaller element is likely already part of the bigger one)
    non_overlapping_elements = []
    for bbox in sorted_clickable_segments:
        if not non_overlapping_elements:
            non_overlapping_elements.append(bbox)
        else:
            overlaps_prior_element = False
            for elem in non_overlapping_elements:
                if is_overlapping(elem,bbox):
                    overlaps_prior_element = True
            if not overlaps_prior_element:
                non_overlapping_elements.append(bbox)
                
    # We can already stop here if we dont have enough elements
    if len(non_overlapping_elements) < n:
        raise ValueError 
    
    # Sort in reverse, largest element first
    non_overlapping_elements_largest_first = sorted(non_overlapping_elements, key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
    
    # Only Select first n
    non_overlapping_elements_largest_first = non_overlapping_elements_largest_first[:n]

    # Segment image into its components
    segments = []
    for bbox in non_overlapping_elements_largest_first:
        segments.append(image.crop((bbox[0],bbox[1],bbox[2],bbox[3])))

    # Create list of normalised coordinates
    normalised_positions = []
    for box in non_overlapping_elements_largest_first:
        x,y = box[0], box[1]
        w,h = box[2]-box[0], box[3]-box[1]
        normalised_positions.append([x/1440.0,y/2560.0,w/1440.0,h/2560.0])
    
    return non_overlapping_elements_largest_first, normalised_positions, segments
            

"""
Segments the UI into its elements, selects 2*n of them, ensuring they arent included in each other,
sorts them by size and returns everything. Raises ValueError if not enough elements can be found.
"""
def get_first_n_sorted_elements_clickable_and_not_clickable(image, image_json, n):
    image = image.convert('RGBA')
    image = image.resize((1440, 2560), Image.Resampling.LANCZOS)
    
    clickable_segments = get_all_bounding_boxes(image_json["activity"]["root"])
    
    # Filter out all 0x0 segments
    clickable_segments = [box for box in clickable_segments if ((box[2]-box[0])*(box[3]-box[1])) > 0]

    # Sort by size ascending
    sorted_clickable_segments = sorted(clickable_segments, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))

    # Filter all elements out that 'cover' at least 80% of some other element (the smaller element is likely already part of the bigger one)
    non_overlapping_elements = []
    for bbox in sorted_clickable_segments:
        if not non_overlapping_elements:
            non_overlapping_elements.append(bbox)
        else:
            overlaps_prior_element = False
            for elem in non_overlapping_elements:
                if is_overlapping(elem,bbox):
                    overlaps_prior_element = True
            if not overlaps_prior_element:
                non_overlapping_elements.append(bbox)
                
    # We can already stop here if we dont have enough elements
    if len(non_overlapping_elements) < n:
        raise ValueError 
    
    # Sort in reverse, largest element first
    non_overlapping_elements_largest_first = sorted(non_overlapping_elements, key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
    
    # Only Select first n
    non_overlapping_elements_largest_first = non_overlapping_elements_largest_first[:n]

    # Segment image into its components
    segments = []
    for bbox in non_overlapping_elements_largest_first:
        segments.append(image.crop((bbox[0],bbox[1],bbox[2],bbox[3])))

    # Create list of normalised coordinates
    normalised_positions = []
    for box in non_overlapping_elements_largest_first:
        x,y = box[0], box[1]
        w,h = box[2]-box[0], box[3]-box[1]
        normalised_positions.append([x/1440.0,y/2560.0,w/1440.0,h/2560.0])
    
    return non_overlapping_elements_largest_first, normalised_positions, segments
            
    
    

    


    

transform_t_to_pil = T.ToPILImage()
transform_to_t = transforms.Compose([transforms.ToTensor()])


def calculate_alignment(boxA, boxB):
    leftern_most_distance = abs(boxA[0] - boxB[0])
    rightern_most_distance = abs(boxA[2] - boxB[2])
    center_most_distance = abs((boxA[2]+boxA[0])/2 - (boxB[2]+boxB[0])/2)

    return min([leftern_most_distance, rightern_most_distance, center_most_distance])    
    
def average_pairwise_alignment(boxes):
    n = len(boxes)
    if n < 2:
        return 0.0  # If there are less than 2 boxes, average IoU is not defined

    alignment_sum = 0
    pair_count = 0
    
    # Iterate over all unique pairs of boxes
    for i in range(n):
        for j in range(i + 1, n):
            alignment = calculate_alignment(boxes[i], boxes[j])
            alignment_sum += alignment
            pair_count += 1

    average_alignment = alignment_sum / pair_count if alignment_sum != 0 else 0.0
    
    return average_alignment



def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    boxA (list): A list [x1, y1, x2, y2] representing the first bounding box.
    boxB (list): A list [x1, y1, x2, y2] representing the second bounding box.
    
    Returns:
    float: IoU value between 0 and 1.
    """

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of the union
    unionArea = boxAArea + boxBArea - interArea

    # Compute the IoU
    iou = interArea / float(unionArea) if unionArea != 0 else 0
    
    return iou

def average_pairwise_iou(boxes):
    """
    Calculate the average pairwise IoU for a list of bounding boxes.
    
    Parameters:
    boxes (list of list): A list of bounding boxes, each represented as [x1, y1, x2, y2].
    
    Returns:
    float: The average IoU for all unique pairs of bounding boxes.
    """
    n = len(boxes)
    if n < 2:
        return 0.0  # If there are less than 2 boxes, average IoU is not defined

    iou_sum = 0
    pair_count = 0
    
    # Iterate over all unique pairs of boxes
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(boxes[i], boxes[j])
            iou_sum += iou
            pair_count += 1

    average_iou = iou_sum / pair_count if pair_count != 0 else 0.0
    
    return average_iou