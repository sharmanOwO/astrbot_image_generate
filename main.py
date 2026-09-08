import asyncio
import base64
import binascii
from typing import Any
from math import gcd

import aiohttp
import astrbot.api.message_components as Comp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr


DEFAULT_API_URL = ""
DEFAULT_EDIT_API_URL = ""
MODEL = "gpt-image-2"
IMAGE_SIZE = "1024x1536"
ALLOWED_IMAGE_SIZES = frozenset({"1024x1024", "1024x1536", "1536x1024"})
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 50 * 1024 * 1024

ImageUpload = tuple[bytes, str, str]


class ImageGenerationError(RuntimeError):
    """An expected error returned while generating or downloading an image."""


@register("image_generate", "local", "使用 gpt-image-2 进行文字生图和图生图", "1.3.1")
class ImageGeneratePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._generation_tasks: set[asyncio.Task[None]] = set()

    @filter.llm_tool(name="generate_image")
    async def generate_image(
        self, event: AstrMessageEvent, prompt: str
    ) -> MessageEventResult:
        """根据用户的文字描述生成图片；当用户要求生成、绘制或创作图片时调用。

        Args:
            prompt(string): 用于生成图片的完整、详细文字描述。
        """
        prompt = prompt.strip()
        if not prompt:
            yield event.plain_result("生图失败：图片描述不能为空。")
            return

        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key:
            yield event.plain_result("请先在插件配置中填写中转站 API Key。")
            return

        task = asyncio.create_task(
            self._generate_and_send(event, prompt, api_key),
            name="image_generate_background_task",
        )
        self._generation_tasks.add(task)
        task.add_done_callback(self._generation_tasks.discard)
        yield event.plain_result("已开始生成图片，完成后会自动发送。")

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str = "original",
    ) -> MessageEventResult:
        """根据用户消息或引用消息中的参考图片进行图生图。

        当用户提供或引用一张或多张图片，并要求修改、重绘、融合图片，
        或根据参考图生成新图片时调用。

        Args:
            prompt(string): 对参考图片的完整修改要求或目标画面描述。
            size(string): 输出图片尺寸。使用 original 采用最接近第一张参考图、
                且宽高均为 16 倍数的尺寸；
                也可以选择 1024x1024、1024x1536、1536x1024，或传入任意一张
                参考图片的实际尺寸，格式为 宽x高。非 16 倍数会自动调整为
                最接近的 16 倍数。
        """
        prompt = prompt.strip()
        if not prompt:
            yield event.plain_result("图生图失败：图片描述不能为空。")
            return

        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key:
            yield event.plain_result("请先在插件配置中填写中转站 API Key。")
            return

        try:
            image_uploads = await self._collect_input_images(event)
        except ImageGenerationError as exc:
            yield event.plain_result(f"图生图失败：{exc}")
            return

        if not image_uploads:
            yield event.plain_result(
                "图生图失败：当前消息或引用消息中没有找到图片。"
            )
            return

        try:
            selected_size = self._resolve_edit_size(size, image_uploads)
        except ImageGenerationError as exc:
            yield event.plain_result(f"图生图失败：{exc}")
            return

        self._start_edit_task(
            event, prompt, api_key, image_uploads, selected_size
        )
        yield event.plain_result(
            f"已开始根据 {len(image_uploads)} 张参考图片生成图片，"
            "完成后会自动发送。"
        )

    @filter.command("tushengtu")
    async def edit_image(
        self, event: AstrMessageEvent, prompt: GreedyStr
    ) -> MessageEventResult:
        """使用当前消息或引用消息中的一张或多张图片进行图生图。"""

        prompt = str(prompt).strip()
        if not prompt:
            yield event.plain_result("图生图失败：图片描述不能为空。")
            return

        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key:
            yield event.plain_result("请先在插件配置中填写中转站 API Key。")
            return

        try:
            image_uploads = await self._collect_input_images(event)
        except ImageGenerationError as exc:
            yield event.plain_result(f"图生图失败：{exc}")
            return

        if not image_uploads:
            yield event.plain_result(
                "图生图失败：请在消息中附带图片，或引用一条包含图片的消息。"
            )
            return

        self._start_edit_task(event, prompt, api_key, image_uploads, IMAGE_SIZE)
        yield event.plain_result(
            f"已开始根据 {len(image_uploads)} 张参考图片生成，完成后会自动发送。"
        )

    def _start_edit_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        api_key: str,
        image_uploads: list[ImageUpload],
        size: str,
    ) -> None:
        task = asyncio.create_task(
            self._edit_and_send(event, prompt, api_key, image_uploads, size),
            name="image_generate_edit_background_task",
        )
        self._generation_tasks.add(task)
        task.add_done_callback(self._generation_tasks.discard)

    async def _generate_and_send(
        self, event: AstrMessageEvent, prompt: str, api_key: str
    ) -> None:
        try:
            image_bytes = await self._generate_image(prompt, api_key)
            await event.send(event.chain_result([Comp.Image.fromBytes(image_bytes)]))
        except asyncio.CancelledError:
            logger.info("image_generate 后台任务已取消")
            raise
        except ImageGenerationError as exc:
            logger.warning("image_generate 请求失败: %s", exc)
            await event.send(event.plain_result(f"生图失败：{exc}"))
        except Exception:
            logger.exception("image_generate 发生未预期错误")
            try:
                await event.send(
                    event.plain_result(
                        "生图失败：插件发生未预期错误，请查看 AstrBot 日志。"
                    )
                )
            except Exception:
                logger.exception("image_generate 无法向用户发送错误消息")

    async def _edit_and_send(
        self,
        event: AstrMessageEvent,
        prompt: str,
        api_key: str,
        image_uploads: list[ImageUpload],
        size: str,
    ) -> None:
        try:
            image_bytes = await self._edit_image(
                prompt, api_key, image_uploads, size
            )
            await event.send(event.chain_result([Comp.Image.fromBytes(image_bytes)]))
        except asyncio.CancelledError:
            logger.info("image_generate 图生图后台任务已取消")
            raise
        except ImageGenerationError as exc:
            logger.warning("image_generate 图生图请求失败: %s", exc)
            await event.send(event.plain_result(f"图生图失败：{exc}"))
        except Exception:
            logger.exception("image_generate 图生图发生未预期错误")
            try:
                await event.send(
                    event.plain_result(
                        "图生图失败：插件发生未预期错误，请查看 AstrBot 日志。"
                    )
                )
            except Exception:
                logger.exception("image_generate 无法向用户发送图生图错误消息")

    async def _generate_image(self, prompt: str, api_key: str) -> bytes:
        api_url = str(self.config.get("api_url", DEFAULT_API_URL)).strip()
        if not api_url.startswith(("http://", "https://")):
            raise ImageGenerationError("插件配置中的中转站地址无效")

        request_timeout = self._request_timeout()
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "size": IMAGE_SIZE,
            "n": 1,
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=request_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    api_url, headers=headers, json=payload
                ) as response:
                    response_text = await response.text()
                    if response.status < 200 or response.status >= 300:
                        detail = self._short_error(response_text)
                        raise ImageGenerationError(
                            f"中转站返回 HTTP {response.status}{detail}"
                        )
                    try:
                        result = await response.json(content_type=None)
                    except (ValueError, TypeError) as exc:
                        raise ImageGenerationError("中转站返回的内容不是有效 JSON") from exc

                image = self._first_image(result)
                if image.get("b64_json"):
                    return self._decode_base64_image(image["b64_json"])

                image_url = image.get("url")
                if image_url:
                    return await self._download_image(session, str(image_url))

                raise ImageGenerationError("中转站响应中没有 b64_json 或图片 URL")
        except ImageGenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise ImageGenerationError("请求中转站超时") from exc
        except aiohttp.ClientError as exc:
            raise ImageGenerationError(f"无法连接中转站：{exc}") from exc

    async def _edit_image(
        self,
        prompt: str,
        api_key: str,
        image_uploads: list[ImageUpload],
        size: str,
    ) -> bytes:
        api_url = str(
            self.config.get("edit_api_url", DEFAULT_EDIT_API_URL)
        ).strip()
        if not api_url.startswith(("http://", "https://")):
            raise ImageGenerationError("插件配置中的图生图接口地址无效")

        form = aiohttp.FormData()
        form.add_field("model", MODEL)
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        form.add_field("n", "1")
        for image_bytes, filename, content_type in image_uploads:
            form.add_field(
                "image",
                image_bytes,
                filename=filename,
                content_type=content_type,
            )

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        timeout = aiohttp.ClientTimeout(total=self._request_timeout())
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    api_url, headers=headers, data=form
                ) as response:
                    response_text = await response.text()
                    if response.status < 200 or response.status >= 300:
                        detail = self._short_error(response_text)
                        raise ImageGenerationError(
                            f"中转站返回 HTTP {response.status}{detail}"
                        )
                    try:
                        result = await response.json(content_type=None)
                    except (ValueError, TypeError) as exc:
                        raise ImageGenerationError(
                            "中转站返回的内容不是有效 JSON"
                        ) from exc

                image = self._first_image(result)
                if image.get("b64_json"):
                    return self._decode_base64_image(image["b64_json"])

                image_url = image.get("url")
                if image_url:
                    return await self._download_image(session, str(image_url))

                raise ImageGenerationError(
                    "中转站响应中没有 b64_json 或图片 URL"
                )
        except ImageGenerationError:
            raise
        except asyncio.TimeoutError as exc:
            raise ImageGenerationError("图生图请求中转站超时") from exc
        except aiohttp.ClientError as exc:
            raise ImageGenerationError(f"无法连接中转站：{exc}") from exc

    async def _collect_input_images(
        self, event: AstrMessageEvent
    ) -> list[ImageUpload]:
        image_components = self._find_image_components(event.get_messages())
        image_uploads: list[ImageUpload] = []
        total_bytes = 0

        for index, component in enumerate(image_components, start=1):
            try:
                encoded = await component.convert_to_base64()
                image_bytes = self._decode_input_image(encoded)
            except ImageGenerationError:
                raise
            except Exception as exc:
                raise ImageGenerationError(
                    f"无法读取第 {index} 张参考图片：{exc}"
                ) from exc

            total_bytes += len(image_bytes)
            if total_bytes > MAX_TOTAL_INPUT_BYTES:
                raise ImageGenerationError("参考图片总大小不能超过 50 MB")

            extension, content_type = self._image_file_type(image_bytes)
            image_uploads.append(
                (image_bytes, f"reference_{index}.{extension}", content_type)
            )

        return image_uploads

    @classmethod
    def _find_image_components(cls, components: list[Any]) -> list[Any]:
        images: list[Any] = []
        for component in components:
            if isinstance(component, Comp.Image):
                images.append(component)
            elif isinstance(component, Comp.Reply) and component.chain:
                images.extend(cls._find_image_components(component.chain))
        return images

    @staticmethod
    def _decode_input_image(value: Any) -> bytes:
        if not isinstance(value, str) or not value:
            raise ImageGenerationError("参考图片 Base64 为空或格式无效")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageGenerationError("参考图片 Base64 无效") from exc
        if not image_bytes:
            raise ImageGenerationError("参考图片内容为空")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ImageGenerationError("单张参考图片不能超过 25 MB")
        return image_bytes

    @staticmethod
    def _image_file_type(image_bytes: bytes) -> tuple[str, str]:
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "jpg", "image/jpeg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png", "image/png"
        if (
            len(image_bytes) >= 12
            and image_bytes.startswith(b"RIFF")
            and image_bytes[8:12] == b"WEBP"
        ):
            return "webp", "image/webp"
        raise ImageGenerationError("参考图片仅支持 JPEG、PNG 或 WebP 格式")

    @classmethod
    def _resolve_edit_size(
        cls, requested_size: str, image_uploads: list[ImageUpload]
    ) -> str:
        source_sizes = [
            cls._normalize_size_to_multiple_of_16(
                cls._image_dimensions(image_bytes)
            )
            for image_bytes, _filename, _content_type in image_uploads
        ]
        allowed_sizes = set(ALLOWED_IMAGE_SIZES)
        allowed_sizes.update(source_sizes)

        normalized = str(requested_size).strip().lower().replace("×", "x")
        if normalized in {"", "original", "source", "原图", "原尺寸"}:
            return source_sizes[0]
        if normalized in allowed_sizes:
            return normalized

        try:
            adjusted_size = cls._normalize_size_to_multiple_of_16(normalized)
        except ImageGenerationError:
            adjusted_size = ""
        if adjusted_size in allowed_sizes:
            return adjusted_size

        choices = "、".join(sorted(allowed_sizes))
        raise ImageGenerationError(
            f"尺寸 {requested_size!r} 不可用；可选择 original 或 {choices}"
        )

    @classmethod
    def _normalize_size_to_multiple_of_16(cls, size: str) -> str:
        normalized = str(size).strip().lower().replace("×", "x")
        parts = normalized.split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ImageGenerationError(f"图片尺寸 {size!r} 格式无效")

        width, height = (int(part) for part in parts)
        if width <= 0 or height <= 0:
            raise ImageGenerationError("图片宽高必须大于 0")
        while  (width * height < 655980):
            width *= 2
            height *= 2
        adjusted_width = cls._nearest_multiple_of_16(width)
        adjusted_height = cls._nearest_multiple_of_16(height)
        return f"{adjusted_width}x{adjusted_height}"

    @staticmethod
    def _nearest_multiple_of_16(value: int) -> int:
        return max(16, ((value + 8) // 16) * 16)

    @classmethod
    def _image_dimensions(cls, image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(image_bytes) < 24:
                raise ImageGenerationError("PNG 参考图片头部不完整")
            width = int.from_bytes(image_bytes[16:20], "big")
            height = int.from_bytes(image_bytes[20:24], "big")
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            width, height = cls._jpeg_dimensions(image_bytes)
        elif (
            len(image_bytes) >= 12
            and image_bytes.startswith(b"RIFF")
            and image_bytes[8:12] == b"WEBP"
        ):
            width, height = cls._webp_dimensions(image_bytes)
        else:
            raise ImageGenerationError("无法识别参考图片尺寸")

        if width <= 0 or height <= 0:
            raise ImageGenerationError("参考图片宽高无效")
        return f"{width}x{height}"

    @staticmethod
    def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int]:
        start_of_frame_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        position = 2
        while position < len(image_bytes):
            if image_bytes[position] != 0xFF:
                position += 1
                continue
            while position < len(image_bytes) and image_bytes[position] == 0xFF:
                position += 1
            if position >= len(image_bytes):
                break

            marker = image_bytes[position]
            position += 1
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if position + 2 > len(image_bytes):
                break

            segment_length = int.from_bytes(
                image_bytes[position : position + 2], "big"
            )
            if segment_length < 2 or position + segment_length > len(image_bytes):
                break
            if marker in start_of_frame_markers and segment_length >= 7:
                height = int.from_bytes(
                    image_bytes[position + 3 : position + 5], "big"
                )
                width = int.from_bytes(
                    image_bytes[position + 5 : position + 7], "big"
                )
                return width, height
            position += segment_length

        raise ImageGenerationError("无法从 JPEG 参考图片读取尺寸")

    @staticmethod
    def _webp_dimensions(image_bytes: bytes) -> tuple[int, int]:
        chunk_type = image_bytes[12:16]
        if chunk_type == b"VP8X" and len(image_bytes) >= 30:
            width = int.from_bytes(image_bytes[24:27], "little") + 1
            height = int.from_bytes(image_bytes[27:30], "little") + 1
            return width, height
        if chunk_type == b"VP8 " and len(image_bytes) >= 30:
            if image_bytes[23:26] != b"\x9d\x01\x2a":
                raise ImageGenerationError("WebP VP8 图片帧头无效")
            width = int.from_bytes(image_bytes[26:28], "little") & 0x3FFF
            height = int.from_bytes(image_bytes[28:30], "little") & 0x3FFF
            return width, height
        if chunk_type == b"VP8L" and len(image_bytes) >= 25:
            if image_bytes[20] != 0x2F:
                raise ImageGenerationError("WebP VP8L 图片帧头无效")
            size_bits = int.from_bytes(image_bytes[21:25], "little")
            width = (size_bits & 0x3FFF) + 1
            height = ((size_bits >> 14) & 0x3FFF) + 1
            return width, height
        raise ImageGenerationError("无法从 WebP 参考图片读取尺寸")

    async def _download_image(
        self, session: aiohttp.ClientSession, image_url: str
    ) -> bytes:
        if not image_url.startswith(("http://", "https://")):
            raise ImageGenerationError("中转站返回了无效的图片 URL")

        try:
            async with session.get(
                image_url, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status < 200 or response.status >= 300:
                    raise ImageGenerationError(
                        f"下载生成图片失败，HTTP {response.status}"
                    )
                image_bytes = await response.read()
        except asyncio.TimeoutError as exc:
            raise ImageGenerationError("下载生成图片超时") from exc
        except aiohttp.ClientError as exc:
            raise ImageGenerationError(f"下载生成图片失败：{exc}") from exc

        self._validate_image_size(image_bytes)
        return image_bytes

    def _request_timeout(self) -> int:
        try:
            timeout = int(self.config.get("request_timeout", 480))
        except (TypeError, ValueError):
            timeout = 480
        return max(30, min(timeout, 600))

    @staticmethod
    def _first_image(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise ImageGenerationError("中转站返回的 JSON 格式无效")
        data = result.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ImageGenerationError("中转站响应中没有有效的图片数据")
        return data[0]

    @classmethod
    def _decode_base64_image(cls, value: Any) -> bytes:
        if not isinstance(value, str) or not value:
            raise ImageGenerationError("中转站返回了无效的 b64_json")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageGenerationError("中转站返回的图片 Base64 无效") from exc
        cls._validate_image_size(image_bytes)
        return image_bytes

    @staticmethod
    def _validate_image_size(image_bytes: bytes) -> None:
        if not image_bytes:
            raise ImageGenerationError("中转站返回了空图片")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ImageGenerationError("中转站返回的图片超过 25 MB")

    @staticmethod
    def _short_error(response_text: str) -> str:
        compact = " ".join(response_text.split())
        return f"：{compact[:300]}" if compact else ""

    async def terminate(self):
        """AstrBot 停用或卸载插件时调用。"""
        tasks = list(self._generation_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._generation_tasks.clear()
