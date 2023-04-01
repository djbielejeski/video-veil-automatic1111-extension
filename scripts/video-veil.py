import copy
import os
import shutil

import cv2
import numpy as np
import gradio as gr
from PIL import Image
import modules.scripts as scripts

from modules import images
from modules.processing import process_images
from modules.shared import opts


def get_all_frames_from_video(video_path: str, count: int = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frame_list: list[np.ndarray] = []
    if not cap.isOpened():
        return
    while True:
        if count is not None and len(frame_list) >= count:
            return frame_list

        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else:
            return frame_list

def get_all_frames_from_folder(folder_path: str, count: int = None) -> list[np.ndarray]:
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

        frame_list.append(np.array(image))

    return frame_list



# TODO: Replace this with ffmpeg export
#def save_gif(path, image_list, name, duration):
#    tmp_dir = path + "/tmp/"
#    if os.path.isdir(tmp_dir):
#        shutil.rmtree(tmp_dir)
#    os.mkdir(tmp_dir)
#    for i, image in enumerate(image_list):
#        images.save_image(image, tmp_dir, f"output_{i}")
#
#    os.makedirs(f"{path}{_BASEDIR}", exist_ok=True)
#
#    image_list[0].save(f"{path}{_BASEDIR}/{name}.gif", save_all=True, append_images=image_list[1:], optimize=False,
#                       duration=duration, loop=0)

color_correction_option_none = 'None'
color_correction_option_video = 'From Source Video'
color_correction_option_generated_image = 'From Stable Diffusion Generated Image'

color_correction_options = [
    color_correction_option_none,
    color_correction_option_video,
    color_correction_option_generated_image,
]

class Script(scripts.Script):

    def title(self):
        return "Video-Veil"

    def show(self, is_img2img):
        return is_img2img

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):

        # Use these later to help guide the user
        # control_net_models_count = opts.data.get("control_net_max_models_num", 1)
        # control_net_allow_script_control = opts.data.get("control_net_allow_script_control", False)
        # p_input_image = get_remote_call(p, "control_net_input_image", None, idx)

        with gr.Group() as video_veil_extension:

            with gr.Row():
                gr.HTML("<br /><h1 style='border-bottom: 1px solid #eee;'>Video Veil</h1>")
            with gr.Row():
                gr.HTML("<div><a href='https://github.com/djbielejeski/video-veil-automatic1111-extension' target='_blank'>Video-Veil Github</a></div>")

            # Input type selection row, allow the user to choose their input type (*.MP4 file or Directory Path)
            with gr.Row():
                with gr.Box():
                    use_images_directory = gr.Checkbox(label=f"Use Directory", value=False, elem_id=f"vv_use_directory_for_video", info="Use Directory of images instead of *.mp4 file")
                    gr.HTML("<br />")

            # Video Uploader Row
            with gr.Row() as video_uploader_row:
                video_path = gr.Video(format='mp4', source='upload', elem_id=f"vv_video_path")
                gr.HTML("<br />")

            # Directory Path Row
            with gr.Row(visible=False) as directory_uploader_row:
                with gr.Box():
                    directory_upload_path = gr.Textbox(
                        label="Directory",
                        value="",
                        elem_id="vv_video_directory",
                        interactive=True,
                        info="Path to directory containing your individual frames for processing."
                    )
                    gr.HTML("<br />")

            # Color Correction
            with gr.Row():
                with gr.Box():
                    color_correction = gr.Dropdown(
                        label="Color Correction",
                        choices=color_correction_options,
                        value=color_correction_option_none,
                        elem_id="vv_color_correction",
                        interactive=True,
                    )

            # Test Processing Row
            with gr.Row():
                with gr.Box():
                    test_run = gr.Checkbox(label=f"Test Run", value=False, elem_id=f"vv_test_run")
                    gr.HTML("<br />")

            with gr.Row(visible=False) as test_run_parameters_row:
                with gr.Box():
                    max_frames_to_test = gr.Slider(
                                    label="# of frames to test",
                                    value=1,
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    elem_id="vv_max_frames_to_test",
                                    interactive=True,
                                )

            # Click handlers and UI Updaters

            # Upload type change
            def change_upload_type_click(
                use_directory_for_video: bool
            ):
                return {
                    video_uploader_row: gr.update(visible=not use_directory_for_video),
                    directory_uploader_row: gr.update(visible=use_directory_for_video),
                }

            use_images_directory.change(
                fn=change_upload_type_click,
                inputs=[
                    use_images_directory
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

            test_run.change(
                fn=test_run_click,
                inputs=[
                    test_run
                ],
                outputs=[
                    test_run_parameters_row
                ]
            )

        return (
            use_images_directory,
            video_path,
            directory_upload_path,
            color_correction,
            test_run,
            max_frames_to_test,
        )

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.
    def run(
            self,
            p,
            use_images_directory: bool,
            video_path: str,
            directory_upload_path: str,
            color_correction: str,
            test_run: bool,
            max_frames_to_test: int,
    ):

        print(f"use_images_directory: {use_images_directory}")


        frames: list[np.ndarray] = []
        frame_count: int = max_frames_to_test if test_run else None

        if use_images_directory:
            print(f"directory_upload_path: {directory_upload_path}")
            if not os.path.exists(directory_upload_path):
                raise Exception(f"Directory not found: '{directory_upload_path}'.")

            frames = get_all_frames_from_folder(folder_path=directory_upload_path, count=frame_count)
        else:
            print(f"video_uploader: {video_path}")
            if not os.path.exists(video_path):
                raise Exception(f"Video not found: '{video_path}'.")
            frames = get_all_frames_from_video(video_path=video_path, count=frame_count)

        print(f"color_correction: {color_correction}")

        print(f"test_run: {test_run}")
        print(f"max_frames_to_test: {max_frames_to_test}")

        print(f"# of frames: {len(frames)}")


        return
