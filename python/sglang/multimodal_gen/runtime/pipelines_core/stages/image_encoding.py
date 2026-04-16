# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import inspect

import numpy as np
import PIL
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    qwen_image_postprocess_text,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.

    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(
        self,
        image_processor,
        image_encoder=None,
        text_encoder=None,
    ) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            text_encoder: An encoder to encode input_ids and pixel values
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def load_model(self):
        if self.server_args.image_encoder_cpu_offload:
            device = get_local_torch_device()
            self.move_to_device(device)

    def offload_model(self):
        if self.server_args.image_encoder_cpu_offload:
            self.move_to_device("cpu")

    def move_to_device(self, device):
        if self.server_args.use_fsdp_inference:
            return
        fields = [
            "image_processor",
            "image_encoder",
        ]
        for field in fields:
            processor = getattr(self, field, None)
            if processor and hasattr(processor, "to"):
                setattr(self, field, processor.to(device))

    def encoding_qwen_image_edit(self, outputs, image_inputs):
        # encoder hidden state
        prompt_embeds = qwen_image_postprocess_text(outputs, image_inputs, 64)
        return prompt_embeds

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode the prompt into image encoder hidden states.
        """

        if batch.condition_image is None:
            return batch
        cuda_device = get_local_torch_device()

        self.load_model()

        image_processor_kwargs = (
            server_args.pipeline_config.prepare_image_processor_kwargs(batch)
        )
        per_prompt_images = image_processor_kwargs.pop("per_prompt_images", None)
        texts = image_processor_kwargs.pop("text", None)

        if per_prompt_images is None:
            per_prompt_images = [batch.condition_image]
            texts = [None] if texts is None else texts

        all_prompt_embeds = []
        all_neg_prompt_embeds = []

        image_processor_call_params = inspect.signature(
            self.image_processor.__call__
        ).parameters
        image_processor_kwargs = {
            k: v
            for k, v in image_processor_kwargs.items()
            if k in image_processor_call_params
        }

        for idx, prompt_images in enumerate(per_prompt_images):
            if not prompt_images:
                continue

            cur_kwargs = image_processor_kwargs.copy()
            if texts and idx < len(texts) and "text" in image_processor_call_params:
                cur_kwargs["text"] = [texts[idx]]

            image_inputs = self.image_processor(
                images=prompt_images, return_tensors="pt", **cur_kwargs
            ).to(cuda_device)

            if self.image_encoder:
                # if an image encoder is provided
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    outputs = self.image_encoder(
                        **image_inputs,
                        **server_args.pipeline_config.image_encoder_extra_args,
                    )
                    image_embeds = server_args.pipeline_config.postprocess_image(
                        outputs
                    )
                batch.image_embeds.append(image_embeds)
            elif self.text_encoder:
                # if a text encoder is provided, e.g. Qwen-Image-Edit
                # 1. neg prompt embeds
                if batch.do_classifier_free_guidance:
                    neg_image_processor_kwargs = (
                        server_args.pipeline_config.prepare_image_processor_kwargs(
                            batch, neg=True
                        )
                    )
                    neg_image_processor_kwargs.pop("per_prompt_images", None)
                    neg_texts = neg_image_processor_kwargs.pop("text", None)
                    if neg_texts and idx < len(neg_texts):
                        neg_image_processor_kwargs["text"] = [neg_texts[idx]]
                    neg_image_inputs = self.image_processor(
                        images=prompt_images,
                        return_tensors="pt",
                        **neg_image_processor_kwargs,
                    ).to(cuda_device)

                with set_forward_context(current_timestep=0, attn_metadata=None):
                    outputs = self.text_encoder(
                        input_ids=image_inputs.input_ids,
                        attention_mask=image_inputs.attention_mask,
                        pixel_values=image_inputs.pixel_values,
                        image_grid_thw=image_inputs.image_grid_thw,
                        output_hidden_states=True,
                    )
                    if batch.do_classifier_free_guidance:
                        neg_outputs = self.text_encoder(
                            input_ids=neg_image_inputs.input_ids,
                            attention_mask=neg_image_inputs.attention_mask,
                            pixel_values=neg_image_inputs.pixel_values,
                            image_grid_thw=neg_image_inputs.image_grid_thw,
                            output_hidden_states=True,
                        )

                all_prompt_embeds.append(
                    self.encoding_qwen_image_edit(outputs, image_inputs)
                )
                if batch.do_classifier_free_guidance:
                    all_neg_prompt_embeds.append(
                        self.encoding_qwen_image_edit(neg_outputs, neg_image_inputs)
                    )

        if all_prompt_embeds:
            batch.prompt_embeds.append(torch.cat(all_prompt_embeds, dim=0))
        if all_neg_prompt_embeds:
            batch.negative_prompt_embeds.append(torch.cat(all_neg_prompt_embeds, dim=0))

        self.offload_model()

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.condition_image=}")
            logger.debug(f"{batch.image_embeds=}")
        result.add_check("pil_image", batch.condition_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        # result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        return result


class LTX2ImageEncodingStage(PipelineStage):
    """Encode ``batch.image_path`` into packed token latents for LTX-2 TI2V.

    This stage runs *before* denoising and populates:
      - ``batch.condition_image`` (resized PIL image)
      - ``batch.image_latent``    (packed [B, S0, D] token latents)
      - ``batch.ltx2_num_image_tokens``
    """

    def __init__(self, vae=None, **kwargs) -> None:
        super().__init__()
        self.vae = vae
        self._condition_image_encoder = None
        self._condition_image_encoder_dir = None

    # ------------------------------------------------------------------
    # Image preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_center_crop(
        img: PIL.Image.Image, *, width: int, height: int
    ) -> PIL.Image.Image:
        return img.resize((width, height), resample=PIL.Image.Resampling.BILINEAR)

    @staticmethod
    def _apply_video_codec_compression(
        img_array: np.ndarray, crf: int = 33
    ) -> np.ndarray:
        """Encode as a single H.264 frame and decode back to simulate compression artifacts."""
        from io import BytesIO

        import av

        if crf == 0:
            return img_array
        height, width = img_array.shape[0] // 2 * 2, img_array.shape[1] // 2 * 2
        img_array = img_array[:height, :width]
        buffer = BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height, stream.width = height, width
        frame = av.VideoFrame.from_ndarray(img_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
        container.close()
        buffer.seek(0)
        container = av.open(buffer)
        decoded = next(container.decode(container.streams.video[0]))
        container.close()
        return decoded.to_ndarray(format="rgb24")

    @staticmethod
    def _resize_center_crop_tensor(
        img: PIL.Image.Image,
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        apply_codec_compression: bool = True,
        codec_crf: int = 33,
    ) -> torch.Tensor:
        """Resize, center-crop, and normalize to [1, C, 1, H, W] tensor in [-1, 1]."""
        import math

        img_array = np.array(img).astype(np.uint8)[..., :3]
        if apply_codec_compression:
            img_array = LTX2ImageEncodingStage._apply_video_codec_compression(
                img_array, crf=codec_crf
            )
        tensor = (
            torch.from_numpy(img_array.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
        )
        src_h, src_w = tensor.shape[2], tensor.shape[3]
        scale = max(height / src_h, width / src_w)
        new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, left = (new_h - height) // 2, (new_w - width) // 2
        tensor = tensor[:, :, top : top + height, left : left + width]
        return ((tensor / 127.5 - 1.0).to(dtype=dtype)).unsqueeze(2)

    # ------------------------------------------------------------------
    # Condition-image encoder (LTX-2.3)
    # ------------------------------------------------------------------

    def _get_condition_image_encoder(
        self,
        server_args: ServerArgs,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        import json
        import os

        from safetensors.torch import load_file as safetensors_load_file

        from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
            LTX23VideoConditionEncoder,
        )

        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir:
            return None

        vae_model_path = server_args.model_paths["vae"]
        encoder_dir = os.path.join(vae_model_path, encoder_subdir)
        config_path = os.path.join(encoder_dir, "config.json")
        weights_path = os.path.join(encoder_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(
                f"LTX-2 condition encoder files not found under {encoder_dir}"
            )

        cached_dir = self._condition_image_encoder_dir
        encoder = self._condition_image_encoder
        if encoder is None or cached_dir != encoder_dir:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            encoder = LTX23VideoConditionEncoder(config)
            encoder.load_state_dict(safetensors_load_file(weights_path), strict=True)
            self._condition_image_encoder = encoder
            self._condition_image_encoder_dir = encoder_dir

        encoder = encoder.to(device=device, dtype=dtype)
        return encoder

    # ------------------------------------------------------------------
    # Stage forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Encode ``batch.image_path`` into packed token latents for LTX-2 TI2V."""
        # Already encoded (e.g. by a prior stage invocation)?
        if (
            batch.image_latent is not None
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        ):
            return batch

        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if batch.image_path is None:
            return batch
        if batch.width is None or batch.height is None:
            raise ValueError("width/height must be provided for LTX-2 TI2V.")
        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 TI2V.")

        from sglang.multimodal_gen.runtime.models.vision_utils import load_image

        image_path = (
            batch.image_path[0]
            if isinstance(batch.image_path, list)
            else batch.image_path
        )

        img = load_image(image_path)
        img_array = np.array(img).astype(np.uint8)[..., :3]
        img_array = self._apply_video_codec_compression(img_array, crf=33)
        conditioned_img = PIL.Image.fromarray(img_array)
        batch.condition_image = self._resize_center_crop(
            conditioned_img, width=int(batch.width), height=int(batch.height)
        )

        latents_device = (
            batch.latents.device
            if isinstance(batch.latents, torch.Tensor)
            else torch.device("cpu")
        )
        encode_dtype = batch.latents.dtype
        original_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            original_dtype != torch.float32
        ) and not server_args.disable_autocast
        condition_image_encoder = self._get_condition_image_encoder(
            server_args, device=latents_device, dtype=encode_dtype
        )
        if condition_image_encoder is None:
            self.vae = self.vae.to(device=latents_device, dtype=encode_dtype)

        video_condition = self._resize_center_crop_tensor(
            conditioned_img,
            width=int(batch.width),
            height=int(batch.height),
            device=latents_device,
            dtype=encode_dtype,
            apply_codec_compression=False,
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=original_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if (
                    condition_image_encoder is None
                    and server_args.pipeline_config.vae_tiling
                ):
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(encode_dtype)

            if condition_image_encoder is not None:
                latent = condition_image_encoder(video_condition)
            else:
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

        if condition_image_encoder is None:
            mode = server_args.pipeline_config.vae_config.encode_sample_mode()
            if mode == "argmax":
                latent = latent_dist.mode()
            elif mode == "sample":
                if batch.generator is None:
                    raise ValueError("Generator must be provided for VAE sampling.")
                latent = latent_dist.sample(batch.generator)
            else:
                raise ValueError(f"Unsupported encode_sample_mode: {mode}")

            # Per-channel normalization: normalized = (x - mean) / std
            mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
            std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
            latent = (latent - mean) / std
        else:
            latent = latent.to(dtype=encode_dtype)

        packed = server_args.pipeline_config.maybe_pack_latents(
            latent, latent.shape[0], batch
        )
        if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
            raise ValueError("Expected packed image latents [B, S0, D].")

        # Fail-fast token count: must match one latent frame's tokens.
        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        latent_h = int(batch.height) // vae_sf
        latent_w = int(batch.width) // vae_sf
        expected_tokens = (latent_h // patch) * (latent_w // patch)
        if int(packed.shape[1]) != int(expected_tokens):
            raise ValueError(
                "LTX-2 conditioning token count mismatch: "
                f"{int(packed.shape[1])=} {int(expected_tokens)=}."
            )

        batch.image_latent = packed
        batch.ltx2_num_image_tokens = int(packed.shape[1])

        if batch.debug:
            logger.info(
                "LTX2 TI2V conditioning prepared: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(batch.image_latent.shape),
                batch.width,
                batch.height,
            )

        if condition_image_encoder is None:
            self.vae.to(original_dtype)
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
            if condition_image_encoder is not None:
                self._condition_image_encoder = condition_image_encoder.to("cpu")

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        return result


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., image_latents).
    """

    def __init__(self, vae: ParallelTiledVAE, **kwargs) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae

    def load_model(self):
        self.vae = self.vae.to(get_local_torch_device())

    def offload_model(self):
        if self.server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel representations into latent space.
        """

        if batch.condition_image is None:
            return batch

        self.load_model()
        num_frames = batch.num_frames

        images = (
            batch.vae_image if batch.vae_image is not None else batch.condition_image
        )
        if not isinstance(images, list):
            images = [images]

        all_image_latents = []
        prepare_condition_image_latent_ids = getattr(
            server_args.pipeline_config, "prepare_condition_image_latent_ids", None
        )
        condition_latents = [] if callable(prepare_condition_image_latent_ids) else None
        for image in images:
            image = self.preprocess(
                image,
            ).to(get_local_torch_device(), dtype=torch.float32)

            # (B, C, H, W) -> (B, C, 1, H, W)
            image = image.unsqueeze(2)

            if num_frames == 1:
                video_condition = image
            else:
                video_condition = torch.cat(
                    [
                        image,
                        image.new_zeros(
                            image.shape[0],
                            image.shape[1],
                            num_frames - 1,
                            image.shape[3],
                            image.shape[4],
                        ),
                    ],
                    dim=2,
                )
            video_condition = video_condition.to(
                device=get_local_torch_device(), dtype=torch.float32
            )

            # Setup VAE precision
            vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            vae_autocast_enabled = (
                vae_dtype != torch.float32
            ) and not server_args.disable_autocast

            # Encode Image
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                # if server_args.vae_sp:
                #     self.vae.enable_parallel()
                if not vae_autocast_enabled:
                    video_condition = video_condition.to(vae_dtype)
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                # for auto_encoder from diffusers
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")

            sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

            latent_condition = self.retrieve_latents(
                latent_dist, generator, sample_mode=sample_mode
            )
            latent_condition = server_args.pipeline_config.postprocess_vae_encode(
                latent_condition, self.vae
            )
            normalized_latent_condition = (
                server_args.pipeline_config.normalize_vae_encode(
                    latent_condition, self.vae
                )
            )
            if normalized_latent_condition is None:
                scaling_factor, shift_factor = (
                    server_args.pipeline_config.get_decode_scale_and_shift(
                        device=latent_condition.device,
                        dtype=latent_condition.dtype,
                        vae=self.vae,
                    )
                )

                # apply shift & scale if needed
                if isinstance(shift_factor, torch.Tensor):
                    shift_factor = shift_factor.to(latent_condition.device)

                if isinstance(scaling_factor, torch.Tensor):
                    scaling_factor = scaling_factor.to(latent_condition.device)

                latent_condition -= shift_factor
                latent_condition = latent_condition * scaling_factor
            else:
                latent_condition = normalized_latent_condition

            if condition_latents is not None:
                condition_latents.append(latent_condition)

            image_latent = server_args.pipeline_config.postprocess_image_latent(
                latent_condition, batch
            )
            all_image_latents.append(image_latent)

        batch.image_latent = torch.cat(all_image_latents, dim=1)
        if condition_latents is not None:
            prepare_condition_image_latent_ids(condition_latents, batch)

        self.offload_model()
        return batch

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def preprocess(
        self,
        image: torch.Tensor | PIL.Image.Image,
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()

        assert batch.condition_image is None or (
            isinstance(batch.condition_image, PIL.Image.Image)
            or isinstance(batch.condition_image, torch.Tensor)
            or isinstance(batch.condition_image, list)
        )
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(batch.num_frames, int)

        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # result.add_check(
        #     "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        # )
        return result
