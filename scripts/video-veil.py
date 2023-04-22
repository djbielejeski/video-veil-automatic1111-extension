import copy

import cv2
import gradio as gr

from scripts.source_video import VideoVeilSourceVideo
from scripts.controlnet_integration import controlnet_external_code

update_cn_script_in_processing = controlnet_external_code.update_cn_script_in_processing

import modules.scripts as scripts
from modules.processing import process_images, setup_color_correction
from modules.shared import opts, cmd_opts, state, sd_model

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

        gr.HTML("<h1 style='border-bottom: 1px solid #eee; margin: 12px 0px 8px !important'>Video Veil</h1>")
        gr.HTML("<div><a style='color: #0969da;' href='https://github.com/djbielejeski/video-veil-automatic1111-extension' target='_blank'>Video-Veil Github</a></div>")

        ebsynth_project_gr = gr.Checkbox(
            label=f"EBSynth Project",
            value=False,
            elem_id=f"vv_ebsynth_project",
            info="EBSynth Project, multiple steps before output",
            # TODO: Turn on when ready
            visible=False,
        )

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

                message = f"<div style='color: #333'>Video Frames found: {first_frame.width}x{first_frame.height}px</div>"
                if temp_video.video_fps > 0:
                    message += f"<div style='color: #333'>Video FPS: {temp_video.video_fps} fps</div>"
                if temp_video.video_total_frames > 0:
                    message += f"<div style='color: #333'>Video Total Frames: {temp_video.video_total_frames} frames</div>"

                return {
                    self.img2img_component: gr.update(value=first_frame.frame_image),
                    self.img2img_w_slider: gr.update(value=first_frame.width),
                    self.img2img_h_slider: gr.update(value=first_frame.height),
                    video_source_info_gr: gr.update(value=message)
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
            ebsynth_project_gr,
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
            ebsynth_project: bool
    ):
        no_video_path = video_path is None or video_path == ""
        no_directory_upload_path = directory_upload_path is None or directory_upload_path == ""
        enabled = not no_video_path or not no_directory_upload_path
        if enabled:
            print(f"use_images_directory: {use_images_directory}")

            video = VideoVeilSourceVideo(
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

            frames_to_process = video.frames_to_process()
            print(f"# of frames to process: {len(frames_to_process)}")

            if len(frames_to_process) > 0:
                state.job_count = len(frames_to_process) * p.n_iter
                state.job_no = 0

                # Pre-process controlnets for speeeeeeeed
                video.preprocess_controlnets(p)

                # Loop over all the frames and process them
                for i, frame in enumerate(frames_to_process):
                    if state.skipped: state.skipped = False
                    if state.interrupted: break

                    # Progress indicator
                    state.job = f"{state.job_no + 1} out of {state.job_count}"

                    cp = copy.copy(p)

                    # Set the ControlNet reference images
                    for cn_index, cn_image in enumerate(frame.controlnet_images):
                        video.controlnet_units[cn_index].image = {'image': cn_image, 'mask': None}

                    # Set the Img2Img reference image to the frame of the video
                    cp.init_images = [frame.frame_image]

                    # Color Correction
                    if color_correction == color_correction_option_none:
                        pass
                    elif color_correction == color_correction_option_video:
                        # Use the source video to apply color correction
                        cp.color_corrections = [setup_color_correction(frame.frame_image)]
                    elif color_correction == color_correction_option_generated_image:
                        if len(frames_to_process) > 0 and frames_to_process[-1].transformed_image is not None:
                            # use the previous frame for color correction
                            cp.color_corrections = [setup_color_correction(frames_to_process[-1].transformed_image)]

                    # Process the image via the normal Img2Img pipeline
                    proc = process_images(cp)

                    # Capture the output, we will use this to re-create our video
                    frame.transformed_image = proc.images[0]

                    cp.close()

                # Show the user what we generated
                proc.images = video.transformed_frames()

                if test_run:
                    for cn_image in video.controlnet_images():
                        proc.images.append(cn_image)

                # now create a video
                if ebsynth_project:
                    video.create_ebsynth_projects(output_directory=cp.outpath_samples)
                else:
                    video.create_mp4(seed=proc.seed, output_directory=cp.outpath_samples, img2img_gallery=self.img2img_gallery)

                video.cleanup()

                del video
                cv2.destroyAllWindows()
        else:
            proc = process_images(p)

        return proc
