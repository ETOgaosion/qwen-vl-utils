from __future__ import annotations

import base64
import math
import warnings
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def _read_video_torchcodec(video_path: str, ele: dict) -> tuple[torch.Tensor, float]:
    """Load video frames as a TCHW tensor and return the source fps.

    Note: ``get_frames_played_in_range`` is half-open ``[start, stop)``, unlike
    torchvision's inclusive ``[start_pts, end_pts]``. Audio stream is dropped.
    """
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise ImportError(
            "torchcodec is required for video processing. Install it with: pip install torchcodec"
        ) from e

    decoder = VideoDecoder(video_path, dimension_order="NCHW")
    md = decoder.metadata

    stream_begin = md.begin_stream_seconds if md.begin_stream_seconds is not None else 0.0
    if md.end_stream_seconds is not None:
        stream_end = md.end_stream_seconds
    elif getattr(md, "duration_seconds", None) is not None:
        stream_end = float(stream_begin) + float(md.duration_seconds)
    else:
        stream_end = None

    start_seconds = ele.get("video_start", stream_begin)
    if start_seconds is None:
        start_seconds = stream_begin
    stop_seconds = ele.get("video_end", stream_end)
    if stop_seconds is None:
        stop_seconds = stream_end

    if stop_seconds is not None and stop_seconds <= start_seconds:
        raise ValueError(
            f"video_end ({stop_seconds}) must be greater than video_start ({start_seconds})."
        )

    if stop_seconds is None:
        # No usable stream bounds: fall back to full-stream slice. decoder[:]
        # honors the constructor-time dimension_order, so returns (T, C, H, W).
        data = decoder[:]
        decoded_seconds = None
    else:
        frame_batch = decoder.get_frames_played_in_range(start_seconds, stop_seconds)
        data = frame_batch.data
        decoded_seconds = float(stop_seconds) - float(start_seconds)

    video_fps = md.average_fps
    if video_fps is None or not (video_fps > 0):
        if decoded_seconds is None and getattr(md, "duration_seconds", None) is not None:
            decoded_seconds = float(md.duration_seconds)
        if decoded_seconds and decoded_seconds > 0 and data.size(0) > 0:
            video_fps = float(data.size(0)) / decoded_seconds
        else:
            raise ValueError(
                "Could not determine source fps: torchcodec returned "
                f"average_fps={md.average_fps!r} and the decoded window has "
                f"{data.size(0)} frames over {decoded_seconds!r} seconds."
            )

    return data, float(video_fps)


def fetch_video(
    ele: dict,
    size_factor: int = FRAME_FACTOR,
    *,
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict] | tuple[torch.Tensor, float] | list[Image.Image] | tuple[list[Image.Image], dict] | tuple[list[Image.Image], float]:
    # image_patch_size: signature parity with upstream / verl; currently ignored
    # because smart_resize uses the hardcoded IMAGE_FACTOR (=14 * SPATIAL_MERGE_SIZE).
    _ = image_patch_size

    if isinstance(ele["video"], str):
        # TODO: support http url

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        video, video_fps = _read_video_torchcodec(video, ele)
        total_frames_before_sample = video.size(0)

        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], size_factor)
        else:
            fps = ele.get("fps", FPS)
            nframes = video.size(0) / video_fps * fps
            nframes = round_by_factor(nframes, size_factor)
            if "min_frames" in ele:
                min_frames = ele["min_frames"]
                if nframes < min_frames:
                    nframes = ceil_by_factor(min_frames, size_factor)
            else:
                min_frames = FPS_MIN_FRAMES
                if nframes < min_frames:
                    warnings.warn(f"nframes is less than DEFAULT_MIN_FRAMES {min_frames}, set to {nframes}.")
                    nframes = ceil_by_factor(min_frames, size_factor)
            if "max_frames" in ele:
                max_frames = ele["max_frames"]
                if nframes > max_frames:
                    nframes = floor_by_factor(max_frames, size_factor)
            else:
                max_frames = FPS_MAX_FRAMES
                if nframes > max_frames:
                    warnings.warn(f"nframes is greater than DEFAULT_MAX_FRAMES {max_frames}, set to {nframes}.")
                    nframes = floor_by_factor(max_frames, size_factor)

        if not (size_factor <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{size_factor}, {video.size(0)}], but got {nframes}.")

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames_before_sample, 1e-6) * float(video_fps)
        height, width = video.shape[2:]
        video = video[idx]

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor), min_pixels * 1.05)
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=size_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        if return_video_metadata:
            src_fps = float(video_fps)
            dur = (
                float(total_frames_before_sample) / src_fps
                if src_fps > 0
                else float(total_frames_before_sample)
            )
            meta = {
                "fps": src_fps,
                "frames_indices": idx.tolist(),
                "total_num_frames": int(total_frames_before_sample),
                "video_backend": "torchcodec",
                "duration": dur,
            }
            if return_video_sample_fps:
                meta["sample_fps"] = float(sample_fps)
            return video, meta
        if return_video_sample_fps:
            return video, float(sample_fps)
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [fetch_image({"image": video_element, **process_info}) for video_element in ele["video"]]
        nframes = ceil_by_factor(len(images), size_factor)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        sample_fps = float(ele.get("fps", FPS))
        # total_num_frames is the padded sampled count (no source video to recover
        # an original frame count from); duration is derived consistently with it.
        meta = {
            "fps": sample_fps,
            "frames_indices": list(range(nframes)),
            "total_num_frames": nframes,
            "video_backend": "frame_list",
            "duration": nframes / sample_fps if sample_fps else 0.0,
        }
        if return_video_metadata:
            if return_video_sample_fps:
                meta["sample_fps"] = float(sample_fps)
            return images, meta
        if return_video_sample_fps:
            return images, float(sample_fps)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    *,
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | tuple[torch.Tensor, dict] | list[Image.Image] | tuple[list[Image.Image], dict]] | None]:
    # Upstream returns a 3-tuple when return_video_kwargs=True; this fork only
    # supports the metadata-driven path, so refuse rather than silently return a
    # 2-tuple that would unpack incorrectly.
    if return_video_kwargs:
        raise NotImplementedError(
            "return_video_kwargs=True is not supported in this fork; pass "
            "return_video_metadata=True and read `sample_fps` from per-video metadata."
        )

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(
                fetch_video(
                    vision_info,
                    image_patch_size=image_patch_size,
                    return_video_metadata=return_video_metadata,
                ),
            )
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
