import copy
import os
import shutil

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import trange
from datetime import datetime, timezone

import modules.scripts as scripts
from modules import processing, shared, script_callbacks, images, sd_samplers, sd_samplers_common
from modules.processing import process_images, setup_color_correction
from modules.shared import opts, cmd_opts, state, sd_model

import torch
import k_diffusion as K

color_correction_option_none = 'None'
color_correction_option_video = 'From Source Video'
color_correction_option_generated_image = 'From Stable Diffusion Generated Image'

color_correction_options = [
    color_correction_option_none,
    color_correction_option_video,
    color_correction_option_generated_image,
]

class Script(scripts.Script):

    def __init__(self):
        self.width: int = 0
        self.height: int = 0

        self.img2img_component = gr.Image()
        self.img2img_gallery = gr.Gallery()
        self.img2img_w_slider = gr.Slider()
        self.img2img_h_slider = gr.Slider()

    def title(self):
        return "Video-Veil"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):

        # Use these later to help guide the user
        # control_net_models_count = opts.data.get("control_net_max_models_num", 1)
        # control_net_allow_script_control = opts.data.get("control_net_allow_script_control", False)

        with gr.Accordion("Video Veil", open=False, elem_id="vv_accordion", visible=is_img2img):

            with gr.Row():
                gr.HTML("<br /><h1 style='border-bottom: 1px solid #eee;'>Video Veil</h1>")
            with gr.Row():
                gr.HTML("<div><a style='color: #0969da;' href='https://github.com/djbielejeski/video-veil-automatic1111-extension' target='_blank'>Video-Veil Github</a></div>")

            # Input type selection row, allow the user to choose their input type (*.MP4 file or Directory Path)
            with gr.Row():
                use_images_directory_gr = gr.Checkbox(label=f"Use Directory", value=False, elem_id=f"vv_use_directory_for_video", info="Use Directory of images instead of *.mp4 file")
                gr.HTML("<br />")


            # Video Uploader Row
            with gr.Row() as video_uploader_row:
                video_path_gr = gr.Video(format='mp4', source='upload', elem_id=f"vv_video_path")
                gr.HTML("<br />")

            # Directory Path Row
            with gr.Row(visible=False) as directory_uploader_row:
                directory_upload_path_gr = gr.Textbox(
                    label="Directory",
                    value="",
                    elem_id="vv_video_directory",
                    interactive=True,
                    info="Path to directory containing your individual frames for processing."
                )
                gr.HTML("<br />")

            # Video Source Info Row
            with gr.Row():
                video_source_info_gr = gr.HTML("")

            # Color Correction
            with gr.Row():
                color_correction_gr = gr.Dropdown(
                    label="Color Correction",
                    choices=color_correction_options,
                    value=color_correction_option_none,
                    elem_id="vv_color_correction",
                    interactive=True,
                )

            # Test Processing Row
            with gr.Row():
                test_run_gr = gr.Checkbox(label=f"Test Run", value=False, elem_id=f"vv_test_run")
                gr.HTML("<br />")

            with gr.Row(visible=False) as test_run_parameters_row:
                max_frames_to_test_gr = gr.Slider(
                                label="# of frames to test",
                                value=1,
                                minimum=1,
                                maximum=100,
                                step=1,
                                elem_id="vv_max_frames_to_test",
                                interactive=True,
                            )

            # Click handlers and UI Updaters

            # If the user selects a video, update the img2img sections
            def video_src_change(
                    use_directory_for_video: bool,
                    video_path: str,
                    directory_upload_path: str,
            ):
                CHECK IF WE HAVE VALID PATHS BEFORE DOING ANY UPDATES HERE WITH THE FRAME

                frames: list[tuple[np.ndarray, Image]] = self.get_frames(
                    use_images_directory=use_directory_for_video,
                    video_path=video_path,
                    directory_upload_path=directory_upload_path,
                    test_run=True,
                    max_frames_to_test=1,
                    throw_errors_when_invalid=False,
                )
                if len(frames) > 0:
                    # Update the img2img settings via the existing Gradio controls
                    frame_array, frame_image = frames[0]

                    self.width, self.height, _ = frame_array.shape
                    return {
                        self.img2img_component: gr.update(value=frame_image),
                        self.img2img_w_slider: gr.update(value=self.width),
                        self.img2img_h_slider: gr.update(value=self.height),
                        video_source_info_gr: gr.update(value=f"<div style='color: #333'>Video Frames found: {self.width}x{self.height}px</div>")
                    }
                else:
                    return {
                        self.img2img_component: gr.update(value=None),
                        self.img2img_w_slider: gr.update(value=512),
                        self.img2img_h_slider: gr.update(value=512),
                        video_source_info_gr: gr.update(value=f"<div style='color: red'>Invalid source, unable to parse video frames from input.</div>")
                    }

            source_inputs = [
                video_path_gr,
                directory_upload_path_gr,
            ]

            for source_input in source_inputs:
                source_input.change(
                    fn=video_src_change,
                    inputs=[
                        use_images_directory_gr,
                        video_path_gr,
                        directory_upload_path_gr,
                    ],
                    outputs=[
                        self.img2img_component,
                        self.img2img_w_slider,
                        self.img2img_h_slider,
                        video_source_info_gr,
                    ]
                )

            # Upload type change
            def change_upload_type_click(
                use_directory_for_video: bool
            ):
                return {
                    video_uploader_row: gr.update(visible=not use_directory_for_video),
                    directory_uploader_row: gr.update(visible=use_directory_for_video),
                }

            use_images_directory_gr.change(
                fn=change_upload_type_click,
                inputs=[
                    use_images_directory_gr
                ],
                outputs=[
                    video_uploader_row,
                    directory_uploader_row
                ]
            )

            # Test run change
            def test_run_click(
                    is_test_run: bool
            ):
                return {
                    test_run_parameters_row: gr.update(visible=is_test_run)
                }

            test_run_gr.change(
                fn=test_run_click,
                inputs=[
                    test_run_gr
                ],
                outputs=[
                    test_run_parameters_row
                ]
            )

        return (
            use_images_directory_gr,
            video_path_gr,
            directory_upload_path_gr,
            color_correction_gr,
            test_run_gr,
            max_frames_to_test_gr,
        )

    # Helper function to get the frames from either the directory or from the uploaded *.mp4 file
    def get_frames(
            self,
            use_images_directory: bool,
            video_path: str,
            directory_upload_path: str,
            test_run: bool,
            max_frames_to_test: int,
            throw_errors_when_invalid: bool = True,
    ) -> list[tuple[np.ndarray, Image]]:

        def get_all_frames_from_video(video_path: str, count: int = None) -> list[tuple[np.ndarray, Image]]:
            cap = cv2.VideoCapture(video_path)
            frame_list: list[tuple[np.ndarray, Image]] = []
            if not cap.isOpened():
                return
            while True:
                if count is not None and len(frame_list) >= count:
                    return frame_list

                ret, frame = cap.read()
                if ret:
                    converted_frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    converted_frame = Image.fromarray(converted_frame_array)
                    frame_list.append([frame, converted_frame])
                else:
                    return frame_list

        def get_all_frames_from_folder(folder_path: str, count: int = None) -> list[tuple[np.ndarray, Image]]:
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_names = [
                file_name for file_name in os.listdir(folder_path)
                if any(file_name.endswith(ext) for ext in image_extensions)
            ]

            if len(image_names) <= 0:
                raise Exception(f"No images (*.png, *.jpg, *.jpeg) found in '{folder_path}'.")

            frame_list: list[np.ndarray] = []
            # Open the images and convert them to np.ndarray
            for i, image_name in enumerate(image_names):

                if count is not None and len(frame_list) >= count:
                    return frame_list

                image_path = os.path.join(folder_path, image_name)

                # Convert the image
                image = Image.open(image_path)

                if not image.mode == "RGB":
                    image = image.convert("RGB")

                frame_list.append([np.array(image), image])

            return frame_list

        frames: list[tuple[np.ndarray, Image]] = []
        frame_count: int = max_frames_to_test if test_run else None

        if use_images_directory:
            print(f"directory_upload_path: {directory_upload_path}")

            if not os.path.exists(directory_upload_path):
                if throw_errors_when_invalid:
                    raise Exception(f"Directory not found: '{directory_upload_path}'.")
            else:
                frames = get_all_frames_from_folder(folder_path=directory_upload_path, count=frame_count)
        else:
            print(f"video_uploader: {video_path}")
            if not os.path.exists(video_path):
                if throw_errors_when_invalid:
                    raise Exception(f"Video not found: '{video_path}'.")
            else:
                frames = get_all_frames_from_video(video_path=video_path, count=frame_count)

        return frames


    def create_mp4(self,
                   use_images_directory: bool,
                   video_path: str,
                   directory_upload_path: str,
                   seed: int,
                   output_directory,
                   output_image_list,
    ) -> str:

        # get the original file name, and slap a timestamp on it
        original_file_name: str = ""
        fps = 30  # TODO: Add this as an option when they pick a folder
        if not use_images_directory:
            original_file_name = os.path.basename(video_path)
            clip = cv2.VideoCapture(video_path)
            if clip:
                fps = clip.get(cv2.CAP_PROP_FPS)
                clip.release()
        else:
            original_file_name = f"{os.path.basename(directory_upload_path)}.mp4"

        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        file_name = f"{date_string}-{seed}-{original_file_name}"

        output_directory = os.path.join(output_directory, "video-veil-output")
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, file_name)

        print(f"Saving *.mp4 to: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

        # write the images to the video file
        for image in output_image_list:
            out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        out.release()

        return output_path

    # From: https://github.com/LonicaMewinsky/gif2gif/blob/main/scripts/gif2gif.py
    # Grab the img2img image components for update later
    # Maybe there's a better way to do this?
    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_image":
            self.img2img_component = component
            return self.img2img_component
        if component.elem_id == "img2img_gallery":
            self.img2img_gallery = component
            return self.img2img_gallery
        if component.elem_id == "img2img_width":
            self.img2img_w_slider = component
            return self.img2img_w_slider
        if component.elem_id == "img2img_height":
            self.img2img_h_slider = component
            return self.img2img_h_slider

    def process(
            self,
            p,
            use_images_directory: bool,
            video_path: str,
            directory_upload_path: str,
            color_correction: str,
            test_run: bool,
            max_frames_to_test: int,
    ):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        no_video_path = video_path is None or video_path == ""
        no_directory_upload_path = directory_upload_path is None or directory_upload_path == ""
        enabled = not no_video_path or not no_directory_upload_path
        if enabled:
            pass
        else:
            print(f"use_images_directory: {use_images_directory}")

            frames: list[tuple[np.ndarray, Image]] = self.get_frames(
                use_images_directory=use_images_directory,
                video_path=video_path,
                directory_upload_path=directory_upload_path,
                test_run=test_run,
                max_frames_to_test=max_frames_to_test,
            )

            print(f"color_correction: {color_correction}")
            print(f"test_run: {test_run}")
            print(f"max_frames_to_test: {max_frames_to_test}")
            print(f"# of frames: {len(frames)}")

            if len(frames) > 0:
                state.job_count = len(frames) * p.n_iter
                state.job_no = 0

                output_image_list = []

                # Loop over all the frames and process them
                for i, frame in enumerate(frames):
                    # TODO: Plumb into Auto1111 progress indicators
                    state.job = f"{state.job_no + 1} out of {state.job_count}"

                    frame_array, frame_image = frame

                    cp = copy.copy(p)

                    # Set the ControlNet reference image
                    cp.control_net_input_image = [frame_array]

                    # Set the Img2Img reference image to the frame of the video
                    cp.init_images = [frame_image]

                    # Color Correction
                    if color_correction == color_correction_option_none:
                        pass
                    elif color_correction == color_correction_option_video:
                        # Use the source video to apply color correction
                        cp.color_corrections = [setup_color_correction(frame_image)]
                    elif color_correction == color_correction_option_generated_image:
                        if len(output_image_list) > 0:
                            # use the previous frame for color correction
                            cp.color_corrections = [setup_color_correction(output_image_list[-1])]


                    # Process the image via the normal Img2Img pipeline
                    proc = process_images(cp)

                    # Capture the output, we will use this to re-create our video
                    img = proc.images[0]
                    output_image_list.append(img)

                    cp.close()

                # Show the user what we generated
                proc.images = output_image_list

                # now create a video
                if not test_run:
                    output_video_file_path = self.create_mp4(
                        use_images_directory=use_images_directory,
                        video_path=video_path,
                        directory_upload_path=directory_upload_path,
                        seed=proc.seed,
                        output_directory=cp.outpath_samples,
                        output_image_list=output_image_list,
                    )

                    self.img2img_gallery.update([output_video_file_path])

        return proc
