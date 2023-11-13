# Zoom In/Out Video Kandinsky

Framework for creating Zoom in / Zoom out video based on inpainting Kandinsky

[kaggle notebook](https://www.kaggle.com/code/ilyaryabov/zoom-in-generative-video)

## Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Additional functionality](#additional-functionality)
- [More examples](#more-examples)

## Installation
```bash
git clone https://github.com/northmen/Zoom_In_Video_Kandinsky
cd Zoom_In_Video_Kandinsky
pip install -r requirements.txt
```

`NOTE:` The code works fine with `torch==2.x`, but when using `torch==1.x` make sure to remove parameter `approximate` in `models/attention.py` file by running
```python
line_number = 325
new_text = "            return F.gelu(gate)"
file = "/home/user/conda/lib/python3.7/site-packages/diffusers/models/attention.py"
!sed -i "$line_number c \\$new_text" "$file"
```
The `line_number` may be differnent depending on `diffusers` version, but for `diffusers==0.21.2` it is `325`

The output of the following command should look like
```bash
!sed -n 325p /home/user/conda/lib/python3.7/site-packages/diffusers/models/attention.py
>>>            return F.gelu(gate)
```
let's mention also that it requires 18735MiB of GPU memory usage 

## Basic usage
This project creates Zoom In or Zoom Out video with square frame of max size `1024 x 1024`, or wide frame of max size `4096 x 1024`
It is the best to use it with some nature or monotonous pictures, and may face some issues drawing animals and people

### Imports
```python
import torch
from classes import ZoomInVideo, WideFrameZoomInVideo
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_prior_emb2emb import KandinskyV22PriorEmb2EmbPipeline
```

First you must `define prior and decoder` for inpainting model.
```
DEVICE = torch.device('cuda')

prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    torch_dtype=torch.float16
)
prior = prior.to(DEVICE)

decoder = KandinskyV22InpaintPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder-inpaint',
    torch_dtype=torch.float16
)
decoder = decoder.to(DEVICE)
```

## Examples
By creating a list of descriptions for key frames you are able to create Zoom In or Zoom out video with square or wide frame
```python
prompt = "autumn forest landscape, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed"
prompts = [prompt] * 8
```

### Square frame video
```python
project_name = "test"
video = ZoomInVideo(project_name,
                    prior=prior,
                    decoder=decoder)
video.set_prompts(prompts)
video.run()
```
https://github.com/northmen/Zoom_In_Video_Kandinsky/assets/32247757/9171f4e1-4fe2-4711-8582-066f2ad7daba


You may find result video in `video.video` path - it has `.avi` extention
To play video in a notebook you may create a gif and play it with:

```python
video.create_gif()
video.play()
```
or just pass argument `create_gif=True` when run your instance, but creating gif is a time consuming operation
```python
video.run(create_gif=True)
video.play()
```

### Wide frame video
```python
project_name = "test"
wide_video = WideFrameZoomInVideo(project_name=project_name,
                                  prior=prior,
                                  decoder=decoder
                                 )
wide_video.set_prompts(prompts)
wide_video.run(create_gif=True)
wide_video.play()
```
https://github.com/northmen/Zoom_In_Video_Kandinsky/assets/32247757/1beb0772-f997-4ea7-9a2a-c75878780e61


or run it without `create_gif` argument to make it faster
```python
wide_video.run()
```
The result video .avi format can be found here `wide_video.video`

[kaggle notebook](https://www.kaggle.com/code/ilyaryabov/zoom-in-generative-video) with example

## Additional functionality
* Set parameter `mode` as `"Out"` or `"In"` when `self.run(mode=mode)` to chose either you want a Zoom In or a Zoom Out video. By default it is set to `"In"`
* You may use an initial image from local by passing `sel.run(init_image=image)`. By default it is set to `None`
* You may check generated key frames with `self.show_key_frames()` function
* In case you don't like some key frames you may repeat their generations with function
  
`self.repeat_frame_generation_starting_from_N(N=frame_number)`

or even set different prompts and then repeat generations

`self.set_prompts()`

`self.repeat_frame_generation_starting_from_N(N=frame_number)`

but then finish the whole pipeline from `self.run()` until `self.gather_frames_into_video()`



## More examples
```python
prompts = [
    "Room in a fantasy elven city, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Room in a fantasy elven city, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "fantasy elven city, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "fantasy elven city, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "fantasy elven forest, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "fantasy elven forest, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon forest, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon forest, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon cyberpunk city, beautiful scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon cyberpunk city, beautiful scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon cosmic flowers, beautiful scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon cosmic flowers, beautiful scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Neon cosmic flowers, beautiful scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
    "Space, stars, neon nebula, breathtaking scenery, vivid colours, bright light, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed",
]

project_name = "example1"
wide_video = WideFrameZoomInVideo(project_name=project_name,
                                  prior=prior,
                                  decoder=decoder,
                                  N=50
                                 )
wide_video.set_prompts(prompts)
wide_video.run(create_gif=True, mode="In")
wide_video.play()
```
https://github.com/northmen/Zoom_In_Video_Kandinsky/assets/32247757/ac312a39-410e-4975-b666-613ffad453e3

```python
prompts = [
    'Sculpture of David in a pose of the Vitruvian Man in the style of Michelangelo',
    'Sculpture of David in a pose of the Vitruvian Man in the style of Michelangelo',
    'Sculpture of David in a pose of the Vitruvian Man in the style of Michelangelo',
    'Sculpture of David in a pose of the Vitruvian Man in the style of Michelangelo',
    'Sculpture of David in a pose of the Vitruvian Man in the style of Michelangelo',
    'Drawing of the Vitruvian Man in the style of Da Vinci',
    'Drawing of the Vitruvian Man in the style of Da Vinci',
    'Drawing of the Vitruvian Man in the style of Da Vinci',
    'Drawing of the Vitruvian Man in the style of Da Vinci',
    'Drawing of the Vitruvian Man in the style of Da Vinci',
    'Painting The Birth of Venus in the style of Botticelli',
    'Painting The Birth of Venus in the style of Botticelli',
    'Painting The Birth of Venus in the style of Botticelli',
    'Painting The Birth of Venus in the style of Botticelli',
    'Painting The Birth of Venus in the style of Botticelli',
    'Furious Ocean in the style of Katsushika Hokusai',
    'Furious Ocean in the style of Katsushika Hokusai',
    'Furious Ocean in the style of Katsushika Hokusai',
    'Furious Ocean in the style of Katsushika Hokusai',
    'Furious Ocean in the style of Katsushika Hokusai',
    'Furious ocean in the style of Van Gogh',
    'Furious ocean in the style of Van Gogh',
    'Furious ocean in the style of Van Gogh',
    'Furious ocean in the style of Van Gogh',
    'Furious ocean in the style of Van Gogh',
    'Morning in a pine forest in the city of Shishkin',
    'Morning in a pine forest in the city of Shishkin',
    'Morning in a pine forest in the city of Shishkin',
    'Morning in a pine forest in the city of Shishkin',
    'Morning in a pine forest in the city of Shishkin'
]

project_name = "example2"
wide_video = WideFrameZoomInVideo(project_name=project_name,
                                  prior=prior,
                                  decoder=decoder
                                 )
wide_video.set_prompts(prompts)
wide_video.run(create_gif=True, mode="In")
wide_video.play()
```
https://github.com/northmen/Zoom_In_Video_Kandinsky/assets/32247757/5400e32e-3c38-4f61-bfde-0a93ac93043d

```python
prompts = [
    "black dog wearing sunglasses",
    "graffiti of a black dog on a blue wall",
    "graffiti on a blue wall",
    "graffiti on a wall",
    "a house with graffiti",
    "a house with graffiti",
    "a house in the forest"
]
project_name = "example3"
video = ZoomInVideo(project_name,
                    prior=prior,
                    decoder=decoder,
                    N=50
                   )
video.set_prompts(prompts)
video.run(create_gif=True, mode="Out")
video.play()
```
https://github.com/northmen/Zoom_In_Video_Kandinsky/assets/32247757/11cb462d-86cf-49ef-8983-7c108046bcda
