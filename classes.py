from pathlib import Path
from IPython.display import Image as play
from functions import *


class ZoomInVideo():
    """
    Square Zoom In/Out Video
    
    example:
    
    prompt = "autumn forest landscape, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed"
    prompts = [prompt] * 8
    
    vidos = ZoomInVideo("Example")
    vidos.set_prompts(prompts)
    vidos.run()
    """
    def __init__(self,
                 project_name,
                 prior,
                 decoder,
                 size=1024, ## size of picture 
                 zoom_out_speed=2./3, ## Zoom out speed
                 N=20, ## Number of frames for interpolation (between 2 neighbouring key frames)
                 num_steps=50 ## number of steps for diffusion
                ):
        self.standart_shape = (size, size)
        self.prompts = None
        self.init_image = None
        self.zoom_out_speed = zoom_out_speed
        self.N = N
        self.num_steps = num_steps
        self.project_name = project_name
        self.prior = prior
        self.decoder = decoder
        self.project_folder = f'video/{project_name}'
        self.key_frames_folder = f'{self.project_folder}/key_frames'
        self.video_frames_folder = f'{self.project_folder}/video_frames'
        Path(self.project_folder).mkdir(parents=True, exist_ok=True)
        Path(self.key_frames_folder).mkdir(parents=True, exist_ok=True)
        Path(self.video_frames_folder).mkdir(parents=True, exist_ok=True)

        
    def clear_key_frames(self):
        """
        Removes all pictures from key_frames_folder
        """
        files = glob.glob(f'{self.key_frames_folder}/*')
        for f in files:
            os.remove(f)

    
    def set_prompts(self, prompts):
        """
        set prompts for key frames
        """
        self.prompts = prompts


    def generate_initial_frame(self):
        """
        Generate the first key frame and saves it 
        to the key_frames_folder
        """
        init_image = generate_image(self.prompts[0], 
                                    self.prior,
                                    self.decoder)
        init_image.save(f"{self.key_frames_folder}/000.jpg")
        self.init_image = init_image


    def set_initial_frame(self, image_path):
        """
        Set the initial frame as a path to a local image
        """
        init_image = Image.open(image_path).resize(self.standart_shape)
        self.init_image = init_image


    def generate_all_key_frames(self):
        """
        Generating all the other key frames (starting from 1)
        based on initial image
        """
        img = self.init_image.copy()
        for i, prompt in tqdm(enumerate(self.prompts[1:])):
            # print(prompt)
            new_frame = extend_square_image(img,
                                            prior=self.prior,
                                            decoder=self.decoder,
                                            prompt=prompt)
            new_frame.save(f"{self.key_frames_folder}/{str(i+1).zfill(3)}.jpg")

            outpainted_cutted_image = zoom_in(new_frame, 1/self.zoom_out_speed)
            corrected_initial_image = correct_initial_image(img, outpainted_cutted_image)

            corrected_initial_image.save(f"{self.key_frames_folder}/{str(i).zfill(3)}.jpg")
            # show_images([img, new_frame])
            img = new_frame.copy()
            
            
    def repeat_frame_generation_starting_from_N(self, N):
        """
        Generating all the other key frames (starting from N)
        based on initial image
        """
        img = Image.open(f"{self.key_frames_folder}/{str(N).zfill(3)}.jpg")
        for i, prompt in tqdm(enumerate(self.prompts[N+1:])):
            new_frame = extend_square_image(img,
                                            prior=self.prior,
                                            decoder=self.decoder,
                                            prompt=prompt)
            new_frame.save(f"{self.key_frames_folder}/{str(N+i+1).zfill(3)}.jpg")
            outpainted_cutted_image = zoom_in(new_frame, 1/self.zoom_out_speed)
            corrected_initial_image = correct_initial_image(img, outpainted_cutted_image)

            corrected_initial_image.save(f"{self.key_frames_folder}/{str(N+i).zfill(3)}.jpg")
            img = new_frame.copy()      
    
    
    def get_key_frames(self):
        """
        search pictures in key_frames_folder
        set self.key_frames as a sorted list of these Images
        """
        key_frames_paths = sorted(glob.glob(f"{self.key_frames_folder}/*"))
        self.key_frames = [Image.open(f) for f in key_frames_paths]
        
        
    def show_key_frames(self, n_rows=1):
        """
        shows key frames of the video
        """
        self.get_key_frames()
        show_images(self.key_frames, n_rows=n_rows)
        
    
    # Interpolating between key frames
    def clear_video_frames_folder(self):
        """
        Removes all images in video_frames_folder
        """
        files = glob.glob(f"{self.video_frames_folder}/*")
        for f in files:
            os.remove(f)
            

    def make_video_frames(self, N=None):
        """
        Interpolating between key frames to create frames for video
        """
        self.clear_video_frames_folder()
        
        if N==None:
            N = self.N
        b = 1. / self.zoom_out_speed
        
        self.get_key_frames()
        key_frames = self.key_frames

        for k in tqdm(range(len(key_frames)-1)):
            frame0 = key_frames[k]
            frame1 = key_frames[k+1]

            for i in range(N):
                zoom = b - (b-1.)*i/N
                img = overlap_two_frames(frame0,
                                         frame1,
                                         zoom_out_speed=self.zoom_out_speed,
                                         alpha=1-i/N
                                        )
                frame = zoom_in(img, zoom)
                frame.save(f"{self.video_frames_folder}/{str(N*k+i).zfill(4)}.jpg")


    def gather_frames_into_video(self):
        """
        Combine all frames in a video .avi
        """
        combine_frames_into_video(self.video_frames_folder,
                                 output=f"{self.project_folder}/{self.project_name}.avi")
        self.video = f"{self.project_folder}/{self.project_name}.avi"


    def create_gif(self, reverse=True):
        """
        Combine all frames in a .gif
        """
        combine_square_frames_into_gif(folder=self.video_frames_folder,
                                        result=f"{self.project_folder}/{self.project_name}.gif",
                                        reverse=reverse)
        self.gif = f"{self.project_folder}/{self.project_name}.gif"


    def play(self):
        """
        show gif in a notebook
        """
        return play(self.gif)


    def run(self, init_image=None, create_gif=False, mode="In"):
        """
        averall algorithm
        you may set initial frame as init_image, 
        otherwise it will be generated from 0th prompt
        creates video .avi format as a result
        """
        VALID_MODES = {"In", "Out"}
        if mode not in VALID_MODES:
            raise ValueError("mode must be one of %r." % VALID_MODES)
        if init_image==None:
            self.generate_initial_frame()
        else:
            self.set_initial_frame(init_image)
        self.generate_all_key_frames()
        self.make_video_frames()
        self.gather_frames_into_video()
        if create_gif:
            if mode == "In":
                reverse = True
            else:
                reverse = False
            self.create_gif(reverse=reverse)



class WideFrameZoomInVideo():
    """
    Wide Frame Zoom In/Out Video
    
    exmaple:
    
    prompt = "autumn forest landscape, photorealistic painting, sharp focus, 8k, perfect composition, trending on artstation, award-winning photograph, unreal engine 5, cinematic smooth, intricate detail, studio photo, highly detailed"
    prompts = [prompt] * 8
    
    wide_vidos = WideFrameZoomInVideo("Example")
    wide_vidos.set_prompts(prompts)
    wide_vidos.run()
    """
    def __init__(self,
                 project_name,
                 prior,
                 decoder,
                 size=1024, ## size of picture 
                 zoom_out_speed=2./3, ## Zoom out speed
                 N=20, ## Number of frames for interpolation (between 2 neighbouring key frames)
                 num_steps=50 ## number of steps for diffusion
                ):
        self.prompts = None
        self.init_image = None
        self.zoom_out_speed = zoom_out_speed
        self.N = N
        self.num_steps = num_steps
        self.project_name = project_name
        self.prior = prior
        self.decoder = decoder
        self.project_folder = f'wide_video/{project_name}'
        self.key_frames_folder = f'{self.project_folder}/key_frames'
        self.video_frames_folder = f'{self.project_folder}/video_frames'
        Path(self.project_folder).mkdir(parents=True, exist_ok=True)
        Path(self.key_frames_folder).mkdir(parents=True, exist_ok=True)
        Path(self.video_frames_folder).mkdir(parents=True, exist_ok=True)


    def clear_key_frames(self):
        """
        Removes all pictures from key_frames_folder
        """
        files = glob.glob(f'{self.key_frames_folder}/*')
        for f in files:
            os.remove(f)


    def set_prompts(self, prompts):
        """
        set prompts for key frames
        """
        self.prompts = prompts


    def generate_initial_frame(self, center_image=None):
        """
        Generate the first key frame and saves it 
        to the key_frames_folder
        """
        if not center_image:
            center_image = generate_image(self.prompts[0],
                                          self.prior,
                                          self.decoder)

        wide_frame, _ = draw_rectangle_frame_based_on_a_center_image(center_image,
                                                                     [self.prompts[0]]*4,
                                                                     self.prior,
                                                                     self.decoder
                                                                    )
        wide_frame.save(f"{self.key_frames_folder}/000.jpg")
        self.init_image = wide_frame
        
        

    def set_initial_frame(self, image_path):
        """
        Set the initial frame as a path to a local image
        """
        init_image = Image.open(image_path)
        self.init_image = init_image


    def generate_all_key_frames(self):
        """
        Generating all the other key frames (starting from 1)
        based on initial image
        """
        self.clear_key_frames()
        self.init_image.save(f"{self.key_frames_folder}/000.jpg")
        
        frame = self.init_image.copy()
        for i, prompt in tqdm(enumerate(self.prompts[1:])):
            new_frame, new_pieces = draw_the_next_frame(frame,
                                                        [prompt]*5,
                                                        self.prior,
                                                        self.decoder)
            new_frame.save(f"{self.key_frames_folder}/{str(i+1).zfill(3)}.jpg")
            frame = new_frame.copy()
        

    def get_key_frames(self):
        """
        search pictures in key_frames_folder
        set self.key_frames as a sorted list of these Images
        """
        key_frames_paths = sorted(glob.glob(f"{self.key_frames_folder}/*"))
        self.key_frames = [Image.open(f) for f in key_frames_paths]


    def repeat_frame_generation_starting_from_N(self, N):
        """
        Generating all the other key frames (starting from N)
        based on initial image
        """
        frame = Image.open(f"{self.key_frames_folder}/{str(N).zfill(3)}.jpg")
        for i, prompt in tqdm(enumerate(self.prompts[N+1:])):
            new_frame, new_pieces = draw_the_next_frame(frame,
                                                        [prompt]*5,
                                                        self.prior,
                                                        self.decoder)
            new_frame.save(f"{self.key_frames_folder}/{str(N+i+1).zfill(3)}.jpg")
            frame = new_frame.copy()

        
    def correct_the_key_frames_borders(self):
        """
        The first part of interpolation
        Corrects the border of each key frame
        This function cuts the border of zoommed {i+1}-th key frame 
        and paste it in {i}-th key frame
        """ 
        self.key_frames = correct_the_border_of_rectangle_frames(self.key_frames)
        for i, frame in enumerate(self.key_frames):
            frame.save(f"{self.key_frames_folder}/{str(i).zfill(3)}.jpg")


    def show_key_frames(self, n_rows=1):
        """
        shows key frames of the video
        """
        self.get_key_frames()
        n_rows = len(self.key_frames)
        show_images(self.key_frames, n_rows=n_rows)

    
    # Interpolating between key frames
    def clear_video_frames_folder(self):
        """
        Removes all images in video_frames_folder
        """
        files = glob.glob(f"{self.video_frames_folder}/*")
        for f in files:
            os.remove(f)
            

    def make_video_frames(self, N=None):
        """
        Interpolation between key frames
        Creates N frames between each pair of 2 neibouring key frames 
        """
        if N==None:
            N = self.N
        b = 1. / self.zoom_out_speed
        
        self.clear_video_frames_folder()

        start = time.time()
        b = 1. / self.zoom_out_speed
        
        result_frames = self.key_frames

        for k in tqdm(range(len(result_frames)-1)):
            frame_0 = result_frames[k]
            frame_1 = result_frames[k+1]

            for i in range(N):
                zoom = b - (b-1.)*i/N
                img = overlap_two_frames_4k(frame_0,
                                            frame_1,
                                            zoom_out_speed=self.zoom_out_speed,
                                            alpha=1-i/N
                                           )
                frame = zoom_in(img, zoom)
                frame.save(f"{self.video_frames_folder}/{str(N*k+i).zfill(4)}.jpg")

        end = time.time()
        print("time:", end - start)

        
    def gather_frames_into_video(self):
        """
        Combine all frames in a video .avi
        """
        combine_frames_into_video(self.video_frames_folder,
                                 output=f"{self.project_folder}/{self.project_name}.avi")
        self.video = f"{self.project_folder}/{self.project_name}.avi"


    def create_gif(self, reverse=True):
        """
        Combine all frames in a .gif
        """
        combine_frames_into_gif(folder=self.video_frames_folder,
                                result=f"{self.project_folder}/{self.project_name}.gif",
                                reverse=reverse)
        self.gif = f"{self.project_folder}/{self.project_name}.gif"


    def play(self):
        """
        show gif in a notebook
        """
        return play(self.gif)
            
    
    def run(self,
            init_image=None,
            create_gif=False,
            center_image=None,
            mode="In"):
        """
        averall algorithm
        you may set initial frame as init_image, 
        otherwise it will be generated from 0th prompt
        """
        VALID_MODES = {"In", "Out"}
        if mode not in VALID_MODES:
            raise ValueError("mode must be one of %r." % VALID_MODES)
        if init_image==None:
            self.generate_initial_frame(center_image=center_image)
        else:
            self.set_initial_frame(init_image)
        self.init_image.save(f"{self.key_frames_folder}/000.jpg")
        self.generate_all_key_frames()
        self.get_key_frames()
        self.correct_the_key_frames_borders()
        self.make_video_frames()
        self.gather_frames_into_video()
        if create_gif:
            if mode == "In":
                reverse = True
            else:
                reverse = False
            self.create_gif(reverse=reverse)