# helper functions for producing different image variations and fusions that combine the grayscale lesion image
# and binary mask of lesion

import cv2
import matplotlib.pyplot as plt
import numpy as np

def convert_to_3_channels(image):
    # Check if image has 2 dimensions (1 channel)
    if image.ndim == 2:
        # Copy the single channel 3 times to create a 3-channel image
        image_3_channels = np.stack((image,)*3, axis=-1)
        return image_3_channels
    # Check if image has 3 dimensions but only 1 channel in the third dimension
    elif image.ndim == 3 and image.shape[2] == 1:
        # Squeeze the third dimension and then copy the channel 3 times
        image_3_channels = np.squeeze(image)
        image_3_channels = np.stack((image_3_channels,)*3, axis=-1)
        return image_3_channels
    # If image already has 3 channels, return it as is
    elif image.ndim == 3 and image.shape[2] == 3:
        return image
    else:
        raise ValueError("Image format not supported")

def resize_and_pad_image(input_image, target_size, padding_value = 0):
    """
    Resizes and pads an image to maintain aspect ratio. Supports integer or tuple target size.
    
    :param input_image: numpy array of the image, grayscale (2D) or RGB (3D).
    :param target_size: int for square target size, or tuple (target_height, target_width) for rectangular target size.
    :param padding_value: value used for padding. Can be a single int or a tuple for RGB.
    :return: Resized and padded image as a numpy array.
    """
    # Return the original image if target_size is None
    if target_size is None:
        scale_factor = 1
        return input_image, scale_factor
        
    # Adjust target size to be a tuple if it's an integer
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    target_height, target_width = target_size
    
    # Determine the current and target aspect ratios
    current_height, current_width = input_image.shape[:2]
    current_aspect_ratio = current_width / current_height
    target_aspect_ratio = target_width / target_height
    
    # Determine the scaling factor and resized dimensions
    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        scale_factor = target_width / current_width
        resized_width = target_width
        resized_height = int(current_height * scale_factor)
    else:
        # Image is taller than target aspect ratio
        scale_factor = target_height / current_height
        resized_height = target_height
        resized_width = int(current_width * scale_factor)
    
    # Resize the image
    resized_image = cv2.resize(input_image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    
    # Calculate padding sizes
    pad_vertical = (target_height - resized_height) // 2
    pad_horizontal = (target_width - resized_width) // 2
    
    # Add padding to the resized image
    if len(input_image.shape) == 3 and isinstance(padding_value, (list, tuple)) and len(padding_value) == 3:
        # For RGB images with RGB padding values
        padded_image = cv2.copyMakeBorder(resized_image, pad_vertical, target_height - resized_height - pad_vertical,
                                          pad_horizontal, target_width - resized_width - pad_horizontal,
                                          cv2.BORDER_CONSTANT, value=padding_value)
    else:
        # For grayscale images or RGB images with single int padding value
        padded_image = cv2.copyMakeBorder(resized_image, pad_vertical, target_height - resized_height - pad_vertical,
                                          pad_horizontal, target_width - resized_width - pad_horizontal,
                                          cv2.BORDER_CONSTANT, value=padding_value)

    return padded_image, scale_factor

#################################################

def plot_images_side_by_side(img1, img2):
    """
    Plots two numpy image arrays side by side, detecting if each is single-channel or three-channel.
    
    :param img1: First numpy image array.
    :param img2: Second numpy image array.
    """
    # Set up the subplot framework
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the first image
    if img1.ndim == 2:  # Single-channel image
        axes[0].imshow(img1, cmap='gray')
    elif img1.ndim == 3 and img1.shape[2] == 3:  # Three-channel image
        axes[0].imshow(img1)
    else:
        raise ValueError("img1 must be either a 2D (grayscale) or 3D (RGB) numpy array.")
    
    # Plot the second image
    if img2.ndim == 2:  # Single-channel image
        axes[1].imshow(img2, cmap='gray')
    elif img2.ndim == 3 and img2.shape[2] == 3:  # Three-channel image
        axes[1].imshow(img2)
    else:
        raise ValueError("img2 must be either a 2D (grayscale) or 3D (RGB) numpy array.")
    
    # Remove the axis for a cleaner look
    #axes[0].axis('off')
    #axes[1].axis('off')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

############################################

def get_largest_contour(mask):
    """Returns contour surrounding largest area in binary mask."""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # fill in straight segments to get more accuracy
    largest_contour = fill_straight_segments(largest_contour)

    return largest_contour

############################################

def fill_straight_segments(contour):
    """Fill in all missing points on straight segments in a contour."""
    filled_contour = []
    for i in range(len(contour)):
        start_point = contour[i][0]
        end_point = contour[(i + 1) % len(contour)][0]  # Loop back to the start for the last segment

        filled_contour.append([start_point])

        # Calculate the differences in the x and y directions
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        max_steps = max(abs(dx), abs(dy))

        # Linear interpolation to fill in all intermediate points
        for step in range(1, max_steps + 1):
            intermediate_point = [
                start_point[0] + step * dx / max_steps,
                start_point[1] + step * dy / max_steps,
            ]
            filled_contour.append([intermediate_point])

    return np.array(filled_contour, dtype=np.int32)

############################################

def create_bfm(img, mask, sigma=20, bdry_only = True):
        
    largest_contour = get_largest_contour(mask)
    
    # Create an empty image and draw the largest contour (not filled)
    contour_mask = np.zeros_like(mask)
    if bdry_only:
        cv2.drawContours(contour_mask, [largest_contour], -1, color=255, thickness=1)
    else:
        cv2.drawContours(contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    
    # Compute the distance map to the contour mask
    dist_to_contour = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 3)

    # Invert the contour mask and compute the distance map to the inverted mask
    inverted_contour_mask = cv2.bitwise_not(contour_mask)
    dist_to_inverted_contour = cv2.distanceTransform(inverted_contour_mask, cv2.DIST_L2, 3)
    
    # Combine the distance maps
    # Adjust the subtraction as needed to reflect the desired distance sign convention
    if bdry_only:
        distance_map = dist_to_inverted_contour - dist_to_contour
    else:
        distance_map = dist_to_inverted_contour
    
    # gaussian transform
    dgtf = np.exp( -distance_map**2/sigma**2 )

    bfm = (img*dgtf).astype(np.uint8)
    
    return bfm, mask

############################################

def add_boundary( img, mask, bdry_color = (255,0,0) ): # red is default color
    """adds the bdry of the mask to the grayscale img for visualization"""
    bdry = get_largest_contour(mask)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 3rd parameter is index of contour to draw, -1 for all
    # 4th parameter is RGB color, defaults to red
    # 5th parameter is thickness
    cv2.drawContours(rgb_image, [bdry], -1, (255, 0, 0), 2)
    return rgb_image

############################################

def makeInputA(img, mask, target_size=None):
    # just resize and pad, does nothing else
    inputA, _ = resize_and_pad_image(img, target_size = target_size, padding_value = 255)
    maskA, _ = resize_and_pad_image(mask, target_size = target_size, padding_value = 0)
    return inputA, maskA

############################################

def makeInputB(img, mask, target_size = None):
    inputB = (img * (mask/255.0) ).astype(np.uint8) # multiply to zero out everything outside the lesion
    inputB[ mask == 0 ] = 255 # convert exterior of lesion to white
    inputB,_ = resize_and_pad_image(inputB, target_size = target_size, padding_value = 255)
    maskB,_ = resize_and_pad_image(mask, target_size = target_size, padding_value = 0)
    return inputB, maskB

############################################

def makeInputC(img, mask, target_size = None):
    inputA,_ = makeInputA(img, mask, target_size=None)
    inputB,_ = makeInputB(img, mask, target_size=None)
    inputC = cv2.merge([inputA,inputB,mask]) # stack three one-channel images
    inputC,_ = resize_and_pad_image(inputC, target_size=target_size,padding_value=(255,255,255))
    maskC,_ = resize_and_pad_image(mask, target_size = target_size, padding_value = 0)
    return inputC, maskC

############################################

def makeInputD(img, mask, target_size=None, pad = 30, pad_value = 255):
    orig_shape = img.shape
    Y,X = orig_shape

    if target_size is None:
        target_size = orig_shape

    # Find the contours of the objects in the mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0][0])
    
    # get coordinates for left, right, top, bottom with padding 30
    # make sure they stay in bounds
    left = max(x-pad,0)
    right = min( x+w+pad, X)
    top = max(y-pad,0)
    bottom = min(y+h+pad,Y)

    # crop the image and the mask
    img = img[top:bottom,left:right]
    mask = mask[top:bottom,left:right]

    inputD, scale_factor = resize_and_pad_image(img, target_size=target_size,padding_value=pad_value)
    maskD, _ = resize_and_pad_image(mask, target_size=target_size, padding_value = 0)

    return inputD, maskD, scale_factor

##############################################

def makeInputE(img, mask, target_size = None, border_area_pct = 20):

    # mask-based crop + resize and pad
    inputD, maskD, scale_factor = makeInputD(img, mask, target_size = target_size)

    # Calculate current area
    current_area = cv2.countNonZero(maskD)
    
    # Calculate target area
    target_area = current_area * (1 + border_area_pct / 100.0)

    # Initialize parameters for dilation
    kernel_size = 3  # Starting kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    maskE = maskD.copy()
    
    # Dilate iteratively until the target area is reached or exceeded
    while cv2.countNonZero(maskE) < target_area:
        maskE = cv2.dilate(maskE, kernel, iterations=1)

    inputE = (inputD * (maskE/255.0)).astype(np.uint8) # multiply to zero out everything outside the lesion
    inputE[ maskE == 0 ] = 255 # convert exterior of lesion to white
    
    return inputE, maskE

###############################################

def makeInputF(img, mask, target_size = None, sigma = 20):
    inputD, maskD,scaling_factor = makeInputD(img, mask, target_size = target_size, pad = 2*sigma, pad_value = 0)
    inputF, maskF = create_bfm(inputD, maskD, sigma = sigma*scaling_factor, bdry_only=True)
    return inputF, maskF

###############################################

def makeInputG(img, mask, target_size = None, sigma = 20):
    inputD, maskD,scaling_factor = makeInputD(img, mask, target_size = target_size, pad = 2*sigma, pad_value = 0)
    inputG, maskG = create_bfm(inputD, maskD, sigma = sigma*scaling_factor, bdry_only=False)
    return inputG, maskG