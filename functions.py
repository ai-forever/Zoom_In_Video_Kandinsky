import os
import cv2
import glob
import time
import numpy as np
import contextlib
from tqdm import tqdm
from kandinsky2 import get_kandinsky2
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops
from IPython.display import Image as play
from pathlib import Path
from matplotlib import pyplot as plt


size = 1024
standart_shape = (size, size)


def combine_square_frames_into_gif(folder="video_frames",
                                    result="result.gif",
                                    reverse=True):
    """
    combine frames from folder to a video with .gif extention
    """
    fp_in = f"{folder}/*.jpg"
    fp_out = result
    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in),
                                    reverse=reverse))
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=1000 // 25, loop=0)
        print(f"file {result} is readyy")


def inpaint_square(init_image, mask, prompt, decoder):
    """
    Inpaint image by a given mask and prompt
    """
    out = decoder(
                prompt=prompt,
                negative_prompt='',
                image=init_image,
                mask_image=mask,
                height=1024,
                width=1024,
                num_inference_steps=75,
                guidance_scale=10
            )
    image = out.images[0]
    return image


def extend_square_image(init_image,
                        prior,
                         decoder,
                         zoom=2./3,
                         ROTATION_SPEED=0, #10,
                         size=size,
                         standart_shape=standart_shape,
                         prompt='',
                         num_steps=100
                        ):
    """
    outpaint picture by a given prompt
    initial image is scaled in the center, empty space on the border will be inpanted by a given prompt
    """
    img = zoom_in(init_image, zoom=zoom)
    mask = get_mask(img)
    # show_images([img,mask])
    extended_image = inpaint(prompt,
                                img,
                                mask,
                                prior,
                                decoder,
                                height=size,
                                width=size,
                                negative_prior_prompt=None
                            )
    return extended_image


def zoom_in(img, zoom):
    """
    center crop of an image by a given zoom
    Can be used for zoom out as well
    """
    w, h = img.size
    x, y = w//2, h//2
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


def show_image(image,
               figsize=(5, 5),
               cmap=None,
               title='',
               xlabel=None,
               ylabel=None,
               axis=False):
    """
    show image with plot
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.show()

    
def show_images(images,
                n_rows=1,
                title='',
                figsize=(15, 15),
                cmap=None,
                xlabel=None,
                ylabel=None,
                axis=False):
    """
    show the list of images
    in several rows
    """
    n_cols = len(images) // n_rows
    if n_rows == n_cols == 1:
        show_image(images[0], title=title, figsize=figsize, cmap=cmap, xlabel=xlabel, ylabel=ylabel, axis=axis)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.tight_layout(pad=0.0)
        axes = axes.flatten()
        for ax, img in zip(axes, images):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis(axis)
        plt.show()
        
        
def overlap_two_frames(frame0, frame1, zoom_out_speed, alpha=1):
    """
    overlap frame1 onto frame0 with a given zoom out parameter
    frame0 will be zoommed out and put onto frame1
    """
    init_image, target_image = frame0, frame1
    im_a = Image.new("L", standart_shape, 0)
    draw = ImageDraw.Draw(im_a)
    bord = size // 10
    draw.rectangle((bord, bord, size-bord, size-bord), fill=int(255*alpha))
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(30))
    im_rgba = init_image.copy()
    im_rgba.putalpha(im_a_blur)
    background = zoom_in(target_image.copy(), 1/zoom_out_speed)
    foreground = im_rgba.copy()
    background.paste(foreground, (0, 0), foreground)
    return background        


## Correcting previous frame after outpainting it
def put_image_in_the_center(image, outpainted_image):
    """
    put image in the center of the outpainted_image
    this function is used for debugging
    """
    border = size // 3
    old_size = (size - border, size - border)
    img = image.copy()
    img = img.resize(old_size)
    new_im = outpainted_image.copy()
    box = tuple((n - o) // 2 for n, o in zip(standart_shape, old_size))
    new_im.paste(img, box)
    zoom_out_img = zoom_in(init_image, 2/3)
    image_gen_n_old_frame = Image.blend(new_im, image_gen, 0)
    return image_gen_n_old_frame


def correct_initial_image(image, outpainted_cutted_image):
    """
    paste the border of outpainted_cutted_image onto image
    first part of color balance interpolation
    """
    im_a = Image.new("L", standart_shape, 0)
    draw = ImageDraw.Draw(im_a)
    bord = size // 10
    draw.rectangle((bord, bord, size-bord, size-bord), fill=int(255))
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(30))
    im_rgba = image.copy()
    im_rgba.putalpha(im_a_blur)
    background = outpainted_cutted_image.copy()
    foreground = im_rgba.copy()
    background.paste(foreground, (0, 0), foreground)
    return background


def generate_image(prompt, prior, decoder):
    """
    generating square image by a given prompt
    """
    img = Image.new("L", standart_shape, 0)
    mask = np.ones(standart_shape)
    picture = inpaint(prompt,
                      img,
                      mask,
                      prior,
                      decoder,
                      height=size,
                      width=size,
                      negative_prior_prompt=None
                    )
    return picture


def combine_frames_into_video(path, output):
    """
    Combine all frames from folder in a video with .avi extention
    """
    frames = sorted(glob.glob(path + "/*"))
    print(len(frames))
    start = time.time()
    frame = cv2.imread(frames[0])
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*'MJPG'),
        25.0,  # fps
        (frame.shape[1], frame.shape[0]),
        isColor=len(frame.shape) > 2)
    for frame in map(cv2.imread, frames):
        writer.write(frame)
    writer.release()
    end = time.time()
    print("time:", end - start)
    print("DONE")
    

def get_mask(img):
    """
    get mask of an image for inpainting
    all black pixels with its surrounding will be detected as empty space
    returns mask
    """
    blur_strength = 1
    M = np.asarray(img.convert('L'))
    M = M.copy()
    M[M>0] = -1
    M += 1
    init_mask = M.copy() * 255
    img = Image.fromarray(init_mask, 'L')
    img = img.filter(ImageFilter.GaussianBlur(blur_strength))
    M = np.asarray(img.convert('L'))
    M = M.copy()
    M[M>0] = 255
    return M


def inpaint(prompt, init_image, mask, prior, decoder, negative_prior_prompt=None, height=512, width=512, strength=1):
    """
    Inpaints picture with a given mask and prompt
    """
    clip_img_emb = prior.interpolate(images_and_prompts=[init_image], weights=[1], num_images_per_prompt=1, ).image_embeds
    prior_output = prior(prompt=prompt,
                         image=clip_img_emb, 
                         strength=strength, 
                         num_inference_steps=25, 
                         num_images_per_prompt=1,)
    if negative_prior_prompt is None:
        prior_output_n = prior_output.negative_image_embeds
    else:
        prior_output_n = prior(prompt=negative_prior_prompt, emb=clip_img_emb, strength=1, num_inference_steps=25, num_images_per_prompt=1,).image_embeds
    out = decoder(
        image_embeds=prior_output.image_embeds,
        negative_image_embeds=prior_output_n,
        image=init_image,
        mask_image=mask,
        height=height,
        width=width,
        num_inference_steps=50,
    )
    image = out.images[0]
    return image


def combine_frames_into_gif(folder="video_frames",
                            result="result.gif",
                            reverse=True):
    """
    Combine wide frames from folder reducing their size by 4 times
    to increase the speed of calculations and memory storage
    output video with .gif format has (1024x256) shape
    thus gif can be displayed in a notebook
    """
    scale = 4
    fp_in = f"{folder}/*.jpg"
    fp_out = result
    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f).resize((4096//scale, 1024//scale)))
                    for f in sorted(glob.glob(fp_in),
                                    reverse=reverse))
        
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=1000 // 25, loop=0)
        print(f"file {result} is readyy")


im_a = Image.new("L", standart_shape, 0)
draw = ImageDraw.Draw(im_a)
bord = size // 8
draw.rectangle((0, 0, size-bord, size), fill=int(255))
im_a_blur_right = im_a.filter(ImageFilter.GaussianBlur(30))

im_a = Image.new("L", standart_shape, 0)
draw = ImageDraw.Draw(im_a)
bord = size // 8
draw.rectangle((bord, 0, size, size), fill=int(255))
im_a_blur_left = im_a.filter(ImageFilter.GaussianBlur(30))


def shift_image_and_draw_right(init_image, prompt, prior, decoder):
    """
    shift the image to the right by three quarters of its width and inpaint it
    """
    image = init_image.copy()
    img_crop = image.crop((size * 3//4, 0, size, size))
    image_shifted = ImageOps.pad(img_crop, image.size, color="black", centering=(0, 0))
    mask_shifted = get_mask(image_shifted)
    new_image = inpaint(
        prompt,
        image_shifted,
        mask_shifted,
        prior,
        decoder,
        height=size,
        width=size,
        negative_prior_prompt=None
    )
    return new_image


def shift_image_and_draw_left(init_image, prompt, prior, decoder):
    """
    shift the image to the left by three quarters of its width and inpaint it
    """
    image = init_image.copy()
    img_crop = image.crop((0, 0, size//4, size))
    image_shifted = ImageOps.pad(img_crop,
                                 image.size,
                                 color="black",
                                 centering=(size*3//4, 0))
    mask_shifted = get_mask(image_shifted)
    new_image = inpaint(
        prompt,
        image_shifted,
        mask_shifted,
        prior,
        decoder,
        height=size,
        width=size,
        negative_prior_prompt=None
    )
    return new_image


def combine_peaces_together(new_peaces):
    """
    combine 5 peaces into 1 wide frame
    """
    new_image_right1 = new_peaces[3]
    new_image_right2 = new_peaces[4]
    new_image_left1 = new_peaces[1]
    new_image_left2 = new_peaces[0]
    
    frame_rectangle = Image.new("RGB", (4*size, size))
    frame_rectangle.paste(new_peaces[2], (int(1.5*size), 0))
    
    new_image_right1.putalpha(im_a_blur_left)
    frame_rectangle.paste(new_image_right1, (int(2.25*size), 0), new_image_right1)
    new_image_right2.putalpha(im_a_blur_left)
    frame_rectangle.paste(new_image_right2, (int(3*size), 0), new_image_right2)
    new_image_left1.putalpha(im_a_blur_right)
    frame_rectangle.paste(new_image_left1, (int(0.75*size), 0), new_image_left1)
    new_image_left2.putalpha(im_a_blur_right)
    frame_rectangle.paste(new_image_left2, (0, 0), new_image_left1)
    
    return frame_rectangle


def draw_rectangle_frame_based_on_a_center_image(img, prompts, prior, decoder):
    """
    create a rectangle frame based on a center image
    two square peaces will be outpainted on the right and two on the left
    """
    new_image_right1 = shift_image_and_draw_right(img, prompts[2], prior, decoder)
    new_image_right2 = shift_image_and_draw_right(new_image_right1, prompts[3], prior, decoder)
    new_image_left1 = shift_image_and_draw_left(img, prompts[1], prior, decoder)
    new_image_left2 = shift_image_and_draw_left(new_image_left1, prompts[0], prior, decoder)
    
    frame_rectangle = Image.new("RGB", (4*size, size))
    frame_rectangle.paste(img, (int(1.5*size), 0))
    
    new_image_right1.putalpha(im_a_blur_left)
    frame_rectangle.paste(new_image_right1, (int(2.25*size), 0), new_image_right1)
    new_image_right2.putalpha(im_a_blur_left)
    frame_rectangle.paste(new_image_right2, (int(3*size), 0), new_image_right2)
    new_image_left1.putalpha(im_a_blur_right)
    frame_rectangle.paste(new_image_left1, (int(0.75*size), 0), new_image_left1)
    new_image_left2.putalpha(im_a_blur_right)
    frame_rectangle.paste(new_image_left2, (0, 0), new_image_left1)
    
    pieces = [new_image_left2, new_image_left1, img, new_image_right1, new_image_right2]
    return frame_rectangle, pieces


def correct_initial_rectangle_image(image, outpainted_cutted_image):
    """
    paste the border of outpainted_cutted_image onto image
    first part of color balance interpolation
    """
    im_a = Image.new("L", (4*size, size), 0)
    draw = ImageDraw.Draw(im_a)
    bord = size // 10
    draw.rectangle((4*bord, bord, 4*(size-bord), size-bord), fill=int(255))
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(30))

    im_rgba = image.copy()
    im_rgba.putalpha(im_a_blur)

    background = outpainted_cutted_image.copy()
    foreground = im_rgba.copy()
    background.paste(foreground, (0, 0), foreground)

    return background


def correct_the_border_of_rectangle_frames(key_frames):
    """
    correct the border of all key frames
    first part of color balance interpolation
    """
    corrected_frames = []
    for i in range(len(key_frames)-1):
        previous_frame = key_frames[i]
        new_frame = key_frames[i+1]

        frame_0 = previous_frame.copy()
        frame_1 = new_frame.copy()

        outpainted_cutted_image = zoom_in(frame_1, 3/2)  #     ###### ZOOM OUT SPEED
        corrected_initial_image = correct_initial_rectangle_image(frame_0,
                                                                  outpainted_cutted_image)
        corrected_frames.append(corrected_initial_image)
        print(f"a frame {i} is corrected")

    corrected_frames.append(key_frames[-1])
    return corrected_frames

        
def overlap_two_frames_4k(frame0, frame1, zoom_out_speed, alpha=1):
    """
    overlap frame1 onto frame0 with a given zoom out parameter
    frame0 will be zoommed out and put onto frame1
    function for images with shape (4096x1024)
    """
    init_image, target_image = frame0.copy(), frame1.copy()

    im_a = Image.new("L", frame0.size, 0)
    draw = ImageDraw.Draw(im_a)
    bord = size // 10
    draw.rectangle((4*bord, bord, 4*(size-bord), size-bord), fill=int(255*alpha))

    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(30))
    im_rgba = init_image.copy()
    im_rgba.putalpha(im_a_blur)
    background = zoom_in(target_image.copy(), 1/zoom_out_speed)
    foreground = im_rgba.copy()
    background.paste(foreground, (0, 0), foreground)
    return background


def draw_the_next_frame(frame0, prompts, prior, decoder):
    """
    draw the next keyframe of a wide-frame video
    frame0 will be zoommed out and outpainted
    """
    new_frame_rectangle = zoom_in(frame0, 2/3)
    
    frame1 = new_frame_rectangle.crop((0, 0, size, size))
    frame2 = new_frame_rectangle.crop((int(0.75*size), 0, int(1.75*size), size))
    frame3 = new_frame_rectangle.crop((int(1.5*size), 0, int(2.5*size), size))
    frame4 = new_frame_rectangle.crop((int(2.25*size), 0, int(3.25*size), size))
    frame5 = new_frame_rectangle.crop((int(3*size), 0, int(4*size), size))
    
    cutted_frames = [frame1,frame2,frame3,frame4,frame5]
    # show_images(cutted_frames)
    masks = [get_mask(frame) for frame in cutted_frames]
    # show_images(masks)
    new_pieces = [
        inpaint(prompt,
                img,
                mask,
                prior,
                decoder,
                height=size,
                width=size,
                negative_prior_prompt=None
               )
        for img,mask, prompt in zip(cutted_frames, masks, prompts)
    ]
    # show_images(new_pieces)
    new_frame = combine_peaces_together(new_pieces)
    return new_frame, new_pieces
