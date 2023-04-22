import os

import copy
import cv2
import math
from PIL import Image
import numpy as np
from datetime import datetime, timezone

from scripts.controlnet_integration import ControlNetUnit, controlnet_external_code, controlnet_HWC3, reverse_preprocessor_aliases, controlnet_preprocessors

class VideoVeilSourceVideoFrame:
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

class VideoVeilGridImage:
    def __init__(
            self,
            frames: list[Image],
            index: int,
            columns: int,
            rows: int,
            frame_width: int,
            frame_height: int,
    ):
        self.frames = copy.copy(frames)
        self.index = index
        self.columns = columns
        self.rows = rows
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.transformed_frames_as_grid: Image = None

        self.grid_width = int(self.frame_width * self.columns)
        self.grid_height = int(self.frame_height * self.rows)
        self.empty_images_to_generate = self.rows * self.columns - len(self.frames)

        self.px_locations = self.get_pixel_locations_for_grid_images()
        self.grid_image: VideoVeilSourceVideoFrame = self.get_grid_of_images()
        self.transformed_grid_image: Image = None

    def get_pixel_locations_for_grid_images(self):
        px_locations = []
        for y in range(self.rows):
            for x in range(self.columns):
                px_locations.append((x * self.frame_width, y * self.frame_height))

        return px_locations

    def get_grid_of_images(self) -> VideoVeilSourceVideoFrame:
        original_frames_as_grid = Image.new(mode="RGB", size=(self.grid_width, self.grid_height))
        for i, frame in enumerate(self.frames):
            x_px, y_px = self.px_locations[i]
            original_frames_as_grid.paste(frame, box=(x_px, y_px))

        return VideoVeilSourceVideoFrame(
            frame_image=original_frames_as_grid,
        )

    def get_transformed_images_from_grid(self) -> list[Image]:
        transformed_frames: list[Image] = []
        if self.transformed_grid_image is not None:
            for i, frame in self.frames:
                left_px, top_px = self.px_locations[i]
                crop_box = (left_px, top_px, left_px + self.frame_width, top_px + self.frame_height)
                temp_image = self.grid_image.copy()
                sub_image = temp_image.crop(crop_box)
                transformed_frames.append(sub_image)

        return transformed_frames

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
            frames_to_grids: bool,
            frames_per_grid: str,
    ):
        self.use_images_directory: bool = use_images_directory
        self.video_path: str = video_path
        self.directory_upload_path: str = directory_upload_path
        self.only_process_every_x_frames = only_process_every_x_frames
        self.test_run: bool = test_run
        self.test_run_frames_count: int = None if not test_run else test_run_frames_count
        self.output_video_path: str = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.video_fps: int = 0
        self.video_total_frames: int = 0
        self.controlnet_units: list[ControlNetUnit] = []
        self.controlnet_modules: list[str] = []
        self.throw_errors_when_invalid = throw_errors_when_invalid

        # Convert frames to grids
        self.frames_to_grids = frames_to_grids

        # frames_per_grid will be "2x2" or "3x3" etc
        self.frames_per_grid = frames_per_grid
        if self.frames_to_grids:
            self.columns, self.rows = [int(i) for i in self.frames_per_grid.split("x")]

            if self.test_run:
                # update the frames count so we can get an entire grid worth of images to test.
                self.test_run_frames_count = self.test_run_frames_count * self.columns * self.rows
                self.only_process_every_x_frames = 1
        else:
            self.columns = 0
            self.rows = 0

        frames = self.get_frames()

        print(f"Frames: {len(frames)}")
        self.frames: list[VideoVeilSourceVideoFrame] = frames

        self.grids_of_frames: list[VideoVeilGridImage] = self.convert_frames_to_grids()
        print(f"Grids: {len(self.grids_of_frames)}")

        if len(self.frames) > 0:
            first_frame = self.frames[0]
            self.video_width, self.video_height = first_frame.width, first_frame.height

    def get_frames(self) -> list[VideoVeilSourceVideoFrame]:
        frames: list[VideoVeilSourceVideoFrame] = []
        if self.use_images_directory:
            print(f"directory_upload_path: {self.directory_upload_path}")
            if self.directory_upload_path is None or not os.path.exists(self.directory_upload_path):
                if self.throw_errors_when_invalid:
                    raise Exception(f"Directory not found: '{self.directory_upload_path}'.")
            else:
                frames = self._load_frames_from_folder()
        else:
            print(f"video_path: {self.video_path}")
            if self.video_path is None or not os.path.exists(self.video_path):
                if self.throw_errors_when_invalid:
                    raise Exception(f"Video not found: '{self.video_path}'.")
            else:
                frames = self._load_frames_from_video()

        return frames

    def convert_frames_to_grids(self) -> list[VideoVeilGridImage]:
        grids_of_frames = []

        if self.frames_to_grids:
            total_frames = len(self.frames)
            if total_frames > 0:
                print(f"Total Frames for Grid: {total_frames}")

                total_images_per_grid = self.columns * self.rows
                total_grids_to_generate = math.ceil(total_frames / total_images_per_grid)

                print(f"Generating {total_grids_to_generate} grids")

                grid_index = 0
                temp_images_for_grid: list[Image] = []
                for i, frame in enumerate(self.frames):
                    if self.test_run and self.test_run_frames_count is not None and len(grids_of_frames) >= self.test_run_frames_count:
                        break

                    is_grid_full: bool = (i % total_images_per_grid) == 0 or (i + 1) == total_frames

                    if is_grid_full and (i > 0 or (total_frames == 1 and i == 0)):
                        grids_of_frames.append(VideoVeilGridImage(
                            frames=temp_images_for_grid,
                            index=grid_index,
                            columns=self.columns,
                            rows=self.rows,
                            frame_width=self.video_width,
                            frame_height=self.video_height,
                        ))
                        temp_images_for_grid = []
                        grid_index += 1

                    temp_images_for_grid.append(frame.frame_image)
            else:
                print("No frames found to generate a grid for")
        else:
            print("Not sending our frames to grids for processing")

        return grids_of_frames

    def frames_to_process(self) -> list[VideoVeilSourceVideoFrame]:
        if self.frames_to_grids:
            grid_images = []
            for grid in self.grids_of_frames:
                grid_images.append(grid.grid_image)
            return grid_images
        else:
            return [
                frame for frame in self.frames
            ]

    def transformed_frames(self) -> list[Image]:
        if self.frames_to_grids:
            transformed_images: list[Image] = []

            grids_with_transforms = [
                frame.transformed_image for frame in self.grids_of_frames
                if frame.transformed_image is not None
            ]

            for grid in grids_with_transforms:
                transformed_images.extend(grid.get_transformed_images_from_grid())

            return transformed_images
        else:
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
                if (cn_unit.enabled is True) and (cn_unit.image is None)
            ]

            print(f"Found {len(self.controlnet_units)} enabled controlnets without input images.")
            for cn_unit in self.controlnet_units:
                # Controlnet: openpose->controlnetPreTrained_openposeDifferenceV1 [1723948e]
                print(f"Controlnet: {cn_unit.module}->{cn_unit.model}")

    def preprocess_controlnets(self, p):
        self._load_controlnets(p)

        if len(self.controlnet_units) > 0:
            print(f"{len(self.controlnet_units)} controlnets are being controlled by the input frames.")
            # TODO: Maybe add a progress indicator here
            frames = self.frames_to_process()
            for cn_unit in self.controlnet_units:
                print(f"Pre-processing controlnet {cn_unit.module} for {len(frames)} frames")
                for i, frame in enumerate(frames):
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
                self.controlnet_modules.append(cn_unit.module)
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

        module_basename: str = reverse_preprocessor_aliases.get(module, module)
        preprocessor = controlnet_preprocessors[module_basename]

        result = None
        if processor_res > 64:
            result, is_image = preprocessor(img, res=processor_res, thr_a=threshold_a, thr_b=threshold_b)
        else:
            result, is_image = preprocessor(img)

        if is_image:
            return result
        else:
            return None

    # TODO
    def create_ebsynth_projects(self, output_directory: str):
        if self.test_run:
            return
        else:
            date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

            # create a project directory
            project_dir = os.path.join(output_directory, "EBSynth", date_string)
            os.makedirs(project_dir, exist_ok=True)

            # create folders holding our transformed images "video_frames_transformed" and our original video images "video_frames_original"
            # make sure the file names are "00000000.png" 8 digits long
            video_frames_transformed = os.path.join(project_dir, "video_frames_transformed")
            os.makedirs(video_frames_transformed, exist_ok=True)

            video_frames_original = os.path.join(project_dir, "video_frames_original")
            os.makedirs(video_frames_original, exist_ok=True)

            # Get all frames from the video
            for i, frame in enumerate(self.frames):
                image_name = "{:08d}.png".format(i)

                if frame.transformed_image is not None:
                    print(f"saving transformed image: {image_name}")
                    transformed_frame_name = os.path.join(video_frames_transformed, image_name)
                    frame.transformed_image.save(transformed_frame_name)

                original_frame_name = os.path.join(video_frames_original, image_name)
                frame.frame_image.save(original_frame_name)

    def create_mp4(self, seed: int, output_directory: str, img2img_gallery=None):
        if self.test_run:
            return
        else:
            # get the original file name, and slap a timestamp on it
            original_file_name: str = ""
            fps = self.video_fps  # TODO: Add this as an option when they pick a folder
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

    def cleanup(self):
        # Reset the controlnet modules
        for i, cn_unit in enumerate(self.controlnet_units):
            cn_unit.module = self.controlnet_modules[i]
            cn_unit.image = None

    def _load_frames_from_folder(self) -> list[VideoVeilSourceVideoFrame]:
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_names = [
            file_name for file_name in os.listdir(self.directory_upload_path)
            if any(file_name.endswith(ext) for ext in image_extensions)
        ]

        if len(image_names) <= 0:
            raise Exception(f"No images (*.png, *.jpg, *.jpeg) found in '{self.directory_upload_path}'.")

        frames: list[VideoVeilSourceVideoFrame] = []
        # Open the images and convert them to np.ndarray
        for i, image_name in enumerate(image_names):

            if self.test_run and self.test_run_frames_count is not None and len(frames) >= self.test_run_frames_count:
                return frames

            if i == 0 or ((i + 1) % self.only_process_every_x_frames == 0):
                image_path = os.path.join(self.directory_upload_path, image_name)

                # Convert the image
                image = Image.open(image_path)

                if not image.mode == "RGB":
                    image = image.convert("RGB")

                frames.append(VideoVeilSourceVideoFrame(frame_image=image))

        return frames

    def _load_frames_from_video(self) -> list[VideoVeilSourceVideoFrame]:
        cap = cv2.VideoCapture(self.video_path)
        frames: list[VideoVeilSourceVideoFrame] = []

        print("loading frames from video")
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        i = 0
        if not cap.isOpened():
            return frames
        while True:
            if self.test_run and self.test_run_frames_count is not None and len(frames) >= self.test_run_frames_count:
                cap.release()
                return frames

            ret, frame = cap.read()
            if ret:
                if i == 0 or ((i + 1) % self.only_process_every_x_frames == 0):
                    frames.append(VideoVeilSourceVideoFrame(frame_array=frame))
            else:
                cap.release()
                return frames

            i += 1

        return frames
