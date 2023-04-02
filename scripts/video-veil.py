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
from modules import processing, shared, images, sd_samplers, sd_samplers_common
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

        with gr.Group() as video_veil_extension:

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
    ) -> list[np.ndarray]:

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

        frames: list[np.ndarray] = []
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

    def find_noise_for_image(self, p, cond, uncond, cfg_scale, steps):
        x = p.init_latent

        s_in = x.new_ones([x.shape[0]])
        dnw = K.external.CompVisDenoiser(shared.sd_model)
        sigmas = dnw.get_sigmas(steps).flip(0)

        state.sampling_steps = steps

        for i in trange(1, len(sigmas)):
            state.sampling_step += 1

            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigmas[i] * s_in] * 2)
            cond_in = torch.cat([uncond, cond])

            image_conditioning = torch.cat([p.image_conditioning] * 2)
            cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}

            c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
            t = dnw.sigma_to_t(sigma_in)

            eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)
            denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

            denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

            d = (x - denoised) / sigmas[i]
            dt = sigmas[i] - sigmas[i - 1]

            x = x + d * dt

            sd_samplers_common.store_latent(x)

            # This shouldn't be necessary, but solved some VRAM issues
            del x_in, sigma_in, cond_in, c_out, c_in, t,
            del eps, denoised_uncond, denoised_cond, denoised, d, dt

        state.nextjob()

        return x / x.std()


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


        frames: list[np.ndarray] = self.get_frames(
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
            state.job_count = len(frames)
            state.job_no = 0

            output_image_list = []
            height, width, _ = frames[0].shape

            # Loop over all the frames and process them
            for i, frame in enumerate(frames):
                # TODO: Plumb into Auto1111 progress indicators
                state.job_no = i

                cp = copy.copy(p)

                # probably should set batch size to 1 here manually
                cp.batch_size = 1

                # overrides for reverse sampling
                cp.sampler_name = "Euler"
                cp.denoising_strength = 1.0

                custom_cfg_scale = 2.0
                original_cfg_scale = cp.cfg_scale
                cp.cfg_scale = custom_cfg_scale

                # Modified from img2imgalt in Automatic1111
                def sample_extra(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
                    shared.state.job_count += 1
                    cond = cp.sd_model.get_learned_conditioning(cp.batch_size * [cp.prompt])
                    uncond = cp.sd_model.get_learned_conditioning(cp.batch_size * [cp.negative_prompt])

                    # this could potentially be
                    # noise = cp.sample(cp, rand_noise, cond, uncond, image_conditioning=cp.image_conditioning)
                    noise = self.find_noise_for_image(cp, cond, uncond, original_cfg_scale, cp.steps)
                    sampler = sd_samplers.create_sampler(cp.sampler_name, cp.sd_model)
                    sigmas = sampler.model_wrap.get_sigmas(cp.steps)
                    noise_dt = noise - (cp.init_latent / sigmas[0])
                    # cp.seed = cp.seed + 1

                    return sampler.sample_img2img(cp, cp.init_latent, noise_dt, conditioning, unconditional_conditioning, image_conditioning=cp.image_conditioning)

                # setup the sampler to use the input image as our noise
                cp.sample = sample_extra

                # set the dimensions on the image
                cp.height = height
                cp.width = width

                # Set the ControlNet reference image
                cp.control_net_input_image = [frame]


                # Set the Img2Img reference image to the frame of the video
                converted_frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                converted_frame = Image.fromarray(converted_frame_array)
                cp.init_images = [converted_frame]

                # Color Correction
                if color_correction == color_correction_option_none:
                    pass
                elif color_correction == color_correction_option_video:
                    # Use the source video to apply color correction
                    cp.color_corrections = [setup_color_correction(converted_frame)]
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

                # get the original file name, and slap a timestamp on it
                original_file_name: str = ""
                fps = 30 # TODO: Add this as an option when they pick a folder
                if not use_images_directory:
                    original_file_name = os.path.basename(video_path)
                    clip = cv2.VideoCapture(video_path)
                    if clip:
                        fps = clip.get(cv2.CAP_PROP_FPS)
                        clip.release()
                else:
                    original_file_name = f"{os.path.basename(directory_upload_path)}.mp4"

                date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
                file_name = f"{date_string}-{proc.seed}-{original_file_name}"

                output_directory = os.path.join(cp.outpath_samples, "video-veil-output")
                os.makedirs(output_directory, exist_ok=True)
                output_path = os.path.join(output_directory, file_name)

                print(f"Saving *.mp4 to: {output_path}")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # write the images to the video file
                for image in output_image_list:
                    out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

                out.release()

        else:
            proc = process_images(p)

        return proc
