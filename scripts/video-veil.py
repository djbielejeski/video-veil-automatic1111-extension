import copy
import os
import shutil

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from datetime import datetime, timezone
import importlib

import modules.scripts as scripts
from modules import processing, shared, script_callbacks, images, sd_samplers, sd_samplers_common
from modules.processing import process_images, setup_color_correction
from modules.shared import opts, cmd_opts, state, sd_model

controlnet_external_code = None
controlnet_global_state = None
controlnet_preprocessors = None
controlnet_HWC3 = None
ControlNetUnit = None
try:
    controlnet_external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    ControlNetUnit = controlnet_external_code.ControlNetUnit
    controlnet_annotator_util = importlib.import_module('extensions.sd-webui-controlnet.annotator.util', 'util')
    controlnet_HWC3 = controlnet_annotator_util.HWC3
    controlnet_global_state = importlib.import_module('extensions.sd-webui-controlnet.scripts.global_state', 'global_state')
    controlnet_preprocessors = controlnet_global_state.cn_preprocessor_modules
except ImportError:
    pass

color_correction_option_none = 'None'
color_correction_option_video = 'From Source Video'
color_correction_option_generated_image = 'From Stable Diffusion Generated Image'

color_correction_options = [
    color_correction_option_none,
    color_correction_option_video,
    color_correction_option_generated_image,
]


class VideoVeilImage:
    def __init__(self, frame_array: np.ndarray = None, frame_image: Image = None):
        self.frame_array = frame_array
        self.frame_image = frame_image
        self.transformed_image: Image = None
        self.controlnet_images: list[np.ndarray] = []

        if frame_array is not None:
            converted_frame_array = cv2.cvtColor(self.frame_array, cv2.COLOR_BGR2RGB)
            self.frame_image = Image.fromarray(converted_frame_array)
        elif frame_image is not None:
            self.frame_array = np.array(self.frame_image)

        # Capture the dimensions of our image
        self.height, self.width, _ = self.frame_array.shape


class VideoVeilSourceVideo:
    def __init__(
            self,
            use_images_directory: bool,
            video_path: str,
            directory_upload_path: str,
            only_process_every_x_frames: int,
            test_run: bool,
            test_run_frames_count: int,
            throw_errors_when_invalid: bool,
    ):
        self.frames: list[VideoVeilImage] = []

        self.use_images_directory: bool = use_images_directory
        self.video_path: str = video_path
        self.directory_upload_path: str = directory_upload_path
        self.only_process_every_x_frames = only_process_every_x_frames
        self.test_run: bool = test_run
        self.test_run_frames_count: int = None if not test_run else test_run_frames_count
        self.output_video_path: str = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.controlnet_units: list[ControlNetUnit] = []


        if use_images_directory:
            print(f"directory_upload_path: {directory_upload_path}")
            if directory_upload_path is None or not os.path.exists(directory_upload_path):
                if throw_errors_when_invalid:
                    raise Exception(f"Directory not found: '{directory_upload_path}'.")
            else:
                self._load_frames_from_folder()
                self._set_video_dimensions()
        else:
            print(f"video_path: {video_path}")
            if video_path is None or not os.path.exists(video_path):
                if throw_errors_when_invalid:
                    raise Exception(f"Video not found: '{video_path}'.")
            else:
                self._load_frames_from_video()
                self._set_video_dimensions()

    def transformed_frames(self) -> list[Image]:
        return [
            frame.transformed_image for frame in self.frames
            if frame.transformed_image is not None
        ]

    def controlnet_images(self) -> list[np.ndarray]:
        cn_images: list[np.ndarray] = []

        for frame in self.frames:
            for cn_image in frame.controlnet_images:
                cn_images.append(cn_image)

        return cn_images

    def _load_controlnets(self, p):
        self.controlnet_units: list[ControlNetUnit] = []
        if controlnet_external_code is not None:
            self.controlnet_units = [
                cn_unit for cn_unit in controlnet_external_code.get_all_units_in_processing(p)
                if cn_unit.enabled
            ]

            print(f"Found {len(self.controlnet_units)} enabled controlnets")
            for cn_unit in self.controlnet_units:
                # Controlnet: openpose[controlnetPreTrained_openposeDifferenceV1 [1723948e]]
                print(f"Controlnet: {cn_unit.module}[{cn_unit.model}]")


    def preprocess_controlnets(self, p):
        self._load_controlnets(p)

        if len(self.controlnet_units) > 0:
            print(f"Pre-processing controlnets. {len(self.controlnet_units)} active.")
            # TODO: Maybe add a progress indicator here
            for cn_unit in self.controlnet_units:
                for i, frame in enumerate(self.frames):
                    # Note: Not capturing masks
                    cn_image: np.ndarray = self._run_controlnet_annotator(
                        image=frame.frame_array,
                        module=cn_unit.module,
                        processor_res=cn_unit.processor_res,
                        threshold_a=cn_unit.threshold_a,
                        threshold_b=cn_unit.threshold_b,
                    )
                    if cn_image is not None:
                        frame.controlnet_images.append(cn_image)

                # turn off the module for the controlnet, since we've already processed the images
                cn_unit.module = "none"

    def _run_controlnet_annotator(
            self,
            image: np.ndarray,
            module: str,
            processor_res: int,
            threshold_a: int,
            threshold_b: int
    ) -> np.ndarray:
        img = controlnet_HWC3(image)
        preprocessor = controlnet_preprocessors[module]
        result = None
        if processor_res > 64:
            result, is_image = preprocessor(img, res=processor_res, thr_a=threshold_a, thr_b=threshold_b)
        else:
            result, is_image = preprocessor(img)

        if is_image:
            return result
        else:
            return None

    def create_mp4(self, seed: int, output_directory: str, img2img_gallery=None):
        if self.test_run:
            return
        else:
            # get the original file name, and slap a timestamp on it
            original_file_name: str = ""
            fps = 30  # TODO: Add this as an option when they pick a folder
            if not self.use_images_directory:
                original_file_name = os.path.basename(self.video_path)
                clip = cv2.VideoCapture(self.video_path)
                if clip:
                    fps = clip.get(cv2.CAP_PROP_FPS)
                    clip.release()
            else:
                original_file_name = f"{os.path.basename(self.directory_upload_path)}.mp4"

            date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            file_name = f"{date_string}-{seed}-{original_file_name}"

            output_directory = os.path.join(output_directory, "video-veil-output")
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, file_name)

            print(f"Saving *.mp4 to: {output_path}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.video_width, self.video_height))

            # write the images to the video file
            for image in self.transformed_frames():
                out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

            out.release()

            self.output_path = output_path
            if img2img_gallery is not None:
                img2img_gallery.update([output_path])

        return

    def _load_frames_from_folder(self):
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_names = [
            file_name for file_name in os.listdir(self.directory_upload_path)
            if any(file_name.endswith(ext) for ext in image_extensions)
        ]

        if len(image_names) <= 0:
            raise Exception(f"No images (*.png, *.jpg, *.jpeg) found in '{self.directory_upload_path}'.")

        self.frames = []
        # Open the images and convert them to np.ndarray
        for i, image_name in enumerate(image_names):

            if self.test_run and self.test_run_frames_count is not None and len(self.frames) >= self.test_run_frames_count:
                return

            if i == 0 or ((i + 1) % self.only_process_every_x_frames == 0):
                image_path = os.path.join(self.directory_upload_path, image_name)

                # Convert the image
                image = Image.open(image_path)

                if not image.mode == "RGB":
                    image = image.convert("RGB")

                self.frames.append(VideoVeilImage(frame_image=image))

        return

    def _load_frames_from_video(self):
        cap = cv2.VideoCapture(self.video_path)
        self.frames = []

        i = 0
        if not cap.isOpened():
            return
        while True:
            if self.test_run and self.test_run_frames_count is not None and len(self.frames) >= self.test_run_frames_count:
                cap.release()
                return

            ret, frame = cap.read()
            if ret:
                if i == 0 or ((i + 1) % self.only_process_every_x_frames == 0):
                    self.frames.append(VideoVeilImage(frame_array=frame))
            else:
                cap.release()
                return

            i += 1

        return

    def _set_video_dimensions(self):
        if len(self.frames) > 0:
            first_frame = self.frames[0]
            self.video_width, self.video_height = first_frame.width, first_frame.height


class Script(scripts.Script):

    def __init__(self):
        self.source_video: VideoVeilSourceVideo = None
        self.img2img_component = gr.Image()
        self.img2img_gallery = gr.Gallery()
        self.img2img_w_slider = gr.Slider()
        self.img2img_h_slider = gr.Slider()

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

        gr.HTML("<h1 style='border-bottom: 1px solid #eee; margin: 12px 0px 8px !important'>Video Veil</h1>")
        gr.HTML("<div><a style='color: #0969da;' href='https://github.com/djbielejeski/video-veil-automatic1111-extension' target='_blank'>Video-Veil Github</a></div>")

        # Input type selection row, allow the user to choose their input type (*.MP4 file or Directory Path)
        use_images_directory_gr = gr.Checkbox(
            label=f"Use Directory",
            value=False,
            elem_id=f"vv_use_directory_for_video",
            info="Use Directory of images instead of *.mp4 file"
        )

        # Spacer
        gr.HTML("<div style='margin: 16px 0px !important; border-bottom: 1px solid #eee;'></div>")

        # Video Uploader Row
        video_path_gr = gr.Video(format='mp4', source='upload', elem_id=f"vv_video_path")

        # Directory Path Row
        directory_upload_path_gr = gr.Textbox(
            label="Directory",
            value="",
            elem_id="vv_video_directory",
            interactive=True,
            visible=False,
            info="Path to directory containing your individual frames for processing."
        )

        # Video Source Info Row
        video_source_info_gr = gr.HTML("")

        gr.HTML("<div style='margin: 16px 0px !important; border-bottom: 1px solid #eee;'></div>")

        with gr.Row():
            with gr.Column():
                # Color Correction
                color_correction_gr = gr.Dropdown(
                    label="Color Correction",
                    choices=color_correction_options,
                    value=color_correction_option_none,
                    elem_id="vv_color_correction",
                    interactive=True,
                )

            with gr.Column():
                # Frame Skip
                only_process_every_x_frames_gr = gr.Slider(
                    label="Only Process every (x) frames",
                    value=1,
                    minimum=1,
                    maximum=100,
                    step=1,
                    elem_id="vv_only_process_every_x_frames",
                    interactive=True,
                )
                gr.HTML("<div style='color: #777; font-size: 12px'>1 will process every frame, 2 will process every other frame, 4 will process every 4th frame, etc</div>")

        gr.HTML("<div style='margin: 16px 0px !important; border-bottom: 1px solid #eee;'></div>")

        # Test Processing Row
        with gr.Row():
            with gr.Column():
                test_run_gr = gr.Checkbox(label=f"Test Run", value=False, elem_id=f"vv_test_run")
            with gr.Column(visible=False) as test_run_parameters_container:
                test_run_frames_count_gr = gr.Slider(
                    label="# of frames to test",
                    value=1,
                    minimum=1,
                    maximum=100,
                    step=1,
                    elem_id="vv_test_run_frames_count",
                    interactive=True,
                )

            # Test run change
            def test_run_click(is_test_run: bool):
                return gr.update(visible=is_test_run)

            test_run_gr.change(fn=test_run_click, inputs=test_run_gr, outputs=test_run_parameters_container)


        # Click handlers and UI Updaters

        # If the user selects a video or directory, update the img2img sections
        def video_src_change(
                use_directory_for_video: bool,
                video_path: str,
                directory_upload_path: str,
        ):
            temp_video = VideoVeilSourceVideo(
                use_images_directory=use_directory_for_video,
                video_path=video_path,
                directory_upload_path=directory_upload_path,
                only_process_every_x_frames=1,
                test_run=True,
                test_run_frames_count=1,
                throw_errors_when_invalid=False
            )

            if len(temp_video.frames) > 0:
                # Update the img2img settings via the existing Gradio controls
                first_frame = temp_video.frames[0]

                return {
                    self.img2img_component: gr.update(value=first_frame.frame_image),
                    self.img2img_w_slider: gr.update(value=first_frame.width),
                    self.img2img_h_slider: gr.update(value=first_frame.height),
                    video_source_info_gr: gr.update(value=f"<div style='color: #333'>Video Frames found: {first_frame.width}x{first_frame.height}px</div>")
                }
            else:
                error_message = "" if video_path is None or directory_upload_path is None or directory_upload_path == "" else "Invalid source, unable to parse video frames from input."
                return {
                    self.img2img_component: gr.update(value=None),
                    self.img2img_w_slider: gr.update(value=512),
                    self.img2img_h_slider: gr.update(value=512),
                    video_source_info_gr: gr.update(value=f"<div style='color: red'>{error_message}</div>")
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
                video_path_gr: gr.update(visible=not use_directory_for_video),
                directory_upload_path_gr: gr.update(visible=use_directory_for_video),
            }

        use_images_directory_gr.change(
            fn=change_upload_type_click,
            inputs=[
                use_images_directory_gr
            ],
            outputs=[
                video_path_gr,
                directory_upload_path_gr
            ]
        )

        return (
            use_images_directory_gr,
            video_path_gr,
            directory_upload_path_gr,
            color_correction_gr,
            only_process_every_x_frames_gr,
            test_run_gr,
            test_run_frames_count_gr,
        )


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

    """
    This function is called if the script has been selected in the script dropdown.
    It must do all processing and return the Processed object with results, same as
    one returned by processing.process_images.

    Usually the processing is done by calling the processing.process_images function.

    args contains all values returned by components from ui()
    """
    def run(
            self,
            p,
            use_images_directory: bool,
            video_path: str,
            directory_upload_path: str,
            color_correction: str,
            only_process_every_x_frames: int,
            test_run: bool,
            test_run_frames_count: int,
    ):
        no_video_path = video_path is None or video_path == ""
        no_directory_upload_path = directory_upload_path is None or directory_upload_path == ""
        enabled = not no_video_path or not no_directory_upload_path
        if enabled:
            print(f"use_images_directory: {use_images_directory}")

            source_video = VideoVeilSourceVideo(
                use_images_directory=use_images_directory,
                video_path=video_path,
                directory_upload_path=directory_upload_path,
                only_process_every_x_frames=only_process_every_x_frames,
                test_run=test_run,
                test_run_frames_count=test_run_frames_count,
                throw_errors_when_invalid=True
            )

            print(f"color_correction: {color_correction}")
            print(f"only_process_every_x_frames: {only_process_every_x_frames}")
            print(f"test_run: {test_run}")
            print(f"test_run_frames_count: {test_run_frames_count}")
            print(f"# of frames: {len(source_video.frames)}")

            if len(source_video.frames) > 0:
                state.job_count = len(source_video.frames) * p.n_iter
                state.job_no = 0

                # Pre-process controlnets for speeeeeeeed
                source_video.preprocess_controlnets(p)

                # Loop over all the frames and process them
                for i, frame in enumerate(source_video.frames):
                    if state.skipped: state.skipped = False
                    if state.interrupted: break

                    # Progress indicator
                    state.job = f"{state.job_no + 1} out of {state.job_count}"

                    cp = copy.copy(p)

                    # Set the ControlNet reference images
                    for cn_index, cn_image in enumerate(frame.controlnet_images):
                        source_video.controlnet_units[cn_index].image = {'image': cn_image, 'mask': None}

                    # Set the Img2Img reference image to the frame of the video
                    cp.init_images = [frame.frame_image]

                    # Color Correction
                    if color_correction == color_correction_option_none:
                        pass
                    elif color_correction == color_correction_option_video:
                        # Use the source video to apply color correction
                        cp.color_corrections = [setup_color_correction(frame.frame_image)]
                    elif color_correction == color_correction_option_generated_image:
                        if len(source_video.frames) > 0 and source_video.frames[-1].transformed_image is not None:
                            # use the previous frame for color correction
                            cp.color_corrections = [setup_color_correction(source_video.frames[-1].transformed_image)]


                    # Process the image via the normal Img2Img pipeline
                    proc = process_images(cp)

                    # Capture the output, we will use this to re-create our video
                    frame.transformed_image = proc.images[0]

                    cp.close()

                # Show the user what we generated
                proc.images = source_video.transformed_frames()

                if test_run:
                    for cn_image in source_video.controlnet_images():
                        proc.images.append(cn_image)

                # now create a video
                source_video.create_mp4(seed=proc.seed, output_directory=cp.outpath_samples, img2img_gallery=self.img2img_gallery)

                del source_video
                cv2.destroyAllWindows()
        else:
            proc = process_images(p)

        return proc
