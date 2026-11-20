"""Multi-image inputs and experimental reference collage for Z-Image.

Avoid SSRF: resolve public hostnames, reject mixed public/private DNS results,
and connect to a validated IP with the original hostname used for HTTPS SNI
and certificate validation (no proxy, no redirects, no environment trust).
"""
import io
import ipaddress
import json
import socket
from pathlib import Path
from urllib.parse import urlsplit

import urllib3
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 4
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageInputError(ValueError):
    def __init__(self, message, status=400, code="invalid_image"):
        super().__init__(message)
        self.status = status
        self.code = code


def parse_urls(urls_text="", url_fields=None):
    result = []
    if urls_text and urls_text.strip():
        try:
            parsed = json.loads(urls_text)
        except (ValueError, TypeError):
            # UI convenience: one URL per line.
            parsed = [x.strip() for x in urls_text.splitlines() if x.strip()]
        if not isinstance(parsed, list) or any(not isinstance(x, str) for x in parsed):
            raise ImageInputError("image_urls 必须是 URL 字符串数组", code="invalid_image_urls")
        result.extend(parsed)
    result.extend(url_fields or [])
    return [v.strip() for v in result if v.strip()]


def _public_target(url):
    try:
        parts = urlsplit(url)
        if (parts.scheme not in ("http", "https") or not parts.hostname or
                parts.username or parts.password or parts.fragment or
                (parts.port and parts.port not in (80, 443))):
            raise ValueError()
        host = parts.hostname.rstrip(".").lower()
        if host in ("localhost",) or host.endswith((".local", ".internal")):
            raise ValueError()
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
        if not addresses or any(not ip.is_global for ip in addresses):
            raise ValueError()
        return parts, host, str(sorted(addresses, key=str)[0]), port
    except (ValueError, OSError) as exc:
        raise ImageInputError("仅允许可公开访问的 HTTP(S) 图片 URL（禁止私有或内网地址）",
                              code="unsafe_image_url") from exc


def fetch_public_image(url):
    parts, host, ip, port = _public_target(url)
    path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
    cls = urllib3.HTTPSConnectionPool if parts.scheme == "https" else urllib3.HTTPConnectionPool
    kwargs = {"host": ip, "port": port, "maxsize": 1, "block": True}
    if parts.scheme == "https":
        kwargs.update(cert_reqs="CERT_REQUIRED", assert_hostname=host, server_hostname=host)
    pool = cls(**kwargs)
    response = None
    try:
        response = pool.request(
            "GET", path,
            headers={"Host": host, "Accept": "image/png,image/jpeg,image/webp,image/gif"},
            retries=False, redirect=False, preload_content=False,
            timeout=urllib3.Timeout(connect=5, read=10),
        )
        if response.status != 200:
            raise ImageInputError("远程图片下载失败（不跟随重定向）",
                                  status=400, code="remote_image_unavailable")
        mime = (response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
        if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ImageInputError("URL 资源不是支持的图片格式", code="invalid_image_type")
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ImageInputError("单张图片不能超过 12 MB", status=413, code="image_too_large")
        return data
    except ImageInputError:
        raise
    except Exception:
        raise ImageInputError("远程图片下载失败或超时", code="remote_image_unavailable") from None
    finally:
        if response is not None:
            response.release_conn()
        pool.close()


def normalized_png(content, filename):
    if len(content) > MAX_BYTES:
        raise ImageInputError("单张图片不能超过 12 MB", status=413, code="image_too_large")
    try:
        with Image.open(io.BytesIO(content)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise ImageInputError("单张图片像素超过 1600 万", status=413,
                                      code="too_many_pixels")
            if img.format not in ("PNG", "JPEG", "WEBP", "GIF"):
                raise ImageInputError("只支持 PNG、JPEG、WebP 和 GIF 图片")
            rendered = ImageOps.exif_transpose(img).convert("RGB")
            rendered.save(filename, "PNG")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ImageInputError("图片格式无效或图片已损坏") from None
    return str(filename)


def prepare_images(contents, urls, folder):
    if not contents and not urls:
        raise ImageInputError("需要至少一张 image 文件或 image_urls", code="missing_image")
    if len(contents) + len(urls) > MAX_IMAGES:
        raise ImageInputError("最多允许 4 张输入图片（含主图）", code="too_many_images")
    paths = []
    # Uploaded files in field order, then URL images in URL order.
    for index, content in enumerate(contents):
        paths.append(normalized_png(content, Path(folder) / f"image_{index}.png"))
    for url in urls:
        index = len(paths)
        paths.append(normalized_png(fetch_public_image(url),
                                    Path(folder) / f"image_{index}.png"))
    return paths


def collage(paths, output):
    """Grid-encode references into ONE img2img input, not native multi-image conditioning."""
    if len(paths) == 1:
        return paths[0], "single_image"
    if not 2 <= len(paths) <= MAX_IMAGES:
        raise ImageInputError("参考图数量不正确")
    side, gutter = 512, 12
    cols, rows = 2, (len(paths) + 1) // 2
    canvas = Image.new("RGB", (cols * side + (cols + 1) * gutter,
                               rows * side + (rows + 1) * gutter), "#f1f1f1")
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), (side, side))
            x = gutter + (i % cols) * (side + gutter)
            y = gutter + (i // cols) * (side + gutter)
            canvas.paste(tile, (x + (side - tile.width) // 2,
                                y + (side - tile.height) // 2))
    canvas.save(output, "PNG")
    return str(output), "experimental_reference_collage"
