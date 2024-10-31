#######################################
#
### Image augmentation originally written for RBE474x
### Repurposed for Terrawarden MQP
#
#######################################


import os
import PIL
import PIL.Image
import torch
import torchvision.transforms as T
import random

directory = '~/Terrawarden/UAVVaste/images' #put directory name here

def random_num(start, stop, step=1):
    rand_num = random.randrange(start, stop, step)
    return rand_num

def resize(img, mask):
    transform = T.Resize(360)
    square_img = transform(img)
    square_mask = transform(mask)
    return square_img, square_mask

# converts image to greyscale
def greyscale(img):
    # converts image to grayscale
    grayscale_transform = T.Grayscale(3)
    grayscaled_image = grayscale_transform(img)

    return grayscaled_image

# rotates image
def rotate(img, mask):
    rotation_angle = random_num(10, 370)
    rotated_img = T.functional.rotate(img, rotation_angle)
    rotated_mask = T.functional.rotate(mask, rotation_angle)
    return rotated_img, rotated_mask

# gaussian blur
def blur(img):
    kernelx = random_num(1, 15, 2)
    kernely = random_num(1, 15, 2)
    blur = T.GaussianBlur(kernel_size = (kernelx, kernely), sigma = (0.1,7))  # values will be changed in final version
    blurred_image = blur(img)
    return blurred_image

# gaussian noise
def addnoise(img): # factor should be random in final
    noise_factor = random_num(1,9)
    noise_factor = noise_factor / 10
    # print(noise_factor)
    img_inputs = T.ToTensor()(img)
    random_tensor = torch.rand_like(img_inputs)
    img_noise = img_inputs + random_tensor * noise_factor
    img_noise = torch.clip(img_noise, 0.1)
    output_image = T.ToPILImage() # not needed in final
    image = output_image(img_noise)
    return image

# color jitter
def jitter(img):

    color_jitter = T.ColorJitter(0.9, 0.9, saturation= (0.1, 1.9), hue= (-0.4, 0.4))
    jittered_image = color_jitter(img)
    return jittered_image

# inverts image
def invert(img):
    invert = T.RandomInvert(p = 1)
    inverted_img = invert(img)
    return inverted_img

# puts image under water
def soup(img):
    soupify = T.ElasticTransform()
    soupified_img = soupify(img)
    return soupified_img
    
def transform_image(filename: str, transform_idx: int):
    img_path = os.path.join(img_directory, filename)
    img = PIL.Image.open(img_path).convert("RGB")
    # mask = PIL.Image.open(mask_path).convert("L")
    img, mask = resize(img, mask)

    if transform_idx == 0:
        max = 0
    elif transform_idx < 4:
        max = 1
        range_min = 1
    elif transform_idx < 7:
        max = 2
        range_min = 2
    elif transform_idx < 9:
        max = 3
        range_min = 2
    else:
        max = 4
        range_min = 2

    i = 0

    while i < max:
        transform = random_num(range_min, 8)
        # print(transform)
        if transform == 1:
            img = greyscale(img)
        if transform == 2:
            img, mask = rotate(img, mask)
        if transform == 3:
            img = blur(img)
        if transform == 4:
            img = addnoise(img)
        if transform == 5:
            img = jitter(img)
        if transform == 6:
            img = invert(img)
        if transform == 7:
            img = soup(img)

        i = i + 1

    img = T.ToTensor()(img)
    # mask = T.ToTensor()(mask)

    return img, mask

    
if __name__ == "__main__":
    img_name = "0363.png"
    background_num = 1
    img_path = f"/home/krsiegall/Terrawarden/UAVVaste/images/"+img_name
    # mask_path = f"/home/krsiegall/Terrawarden/UAVVaste/images/"+img_name
    img = PIL.Image.open(img_path).convert("RGB")
    # img = greyscale(img)
    # img, mask = rotate(img, mask)
    img = blur(img)
    img = addnoise(img)
    img = jitter(img)
    # img = invert(img)
    img = soup(img)
    img.save("/home/krsiegall/Terrawarden/UAVVaste/fuckmeup.png")