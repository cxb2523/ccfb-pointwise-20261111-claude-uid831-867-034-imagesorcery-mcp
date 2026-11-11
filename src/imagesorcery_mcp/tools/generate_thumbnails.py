import os
from typing import Annotated, Optional

from fastmcp import FastMCP
from PIL import Image
from pydantic import Field

# Import the central logger
from imagesorcery_mcp.logging_config import logger

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

FORMAT_ALIASES = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "bmp": "BMP",
    "gif": "GIF",
    "tif": "TIFF",
    "tiff": "TIFF",
    "webp": "WEBP",
}

FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


def _is_within(path: str, root: str) -> bool:
    """Check that ``path`` stays inside ``root`` using commonpath.

    Both arguments must already be normalized real/absolute paths.
    Returns False on ValueError (e.g. different drives on Windows).
    """
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:
        return False


def _collect_files(input_dir: str, root_real: str, failed: list) -> list:
    """Recursively collect (logical_path, real_path) file entries under input_dir.

    Logical paths keep the directory layout as seen from input_dir (symlink
    names preserved) so output mirroring does not flatten linked directories.
    Real paths are used for containment checks and decoding. Directories that
    escape root_real or form symlink cycles are recorded in ``failed``.
    """
    entries = []
    visited = {os.path.normcase(root_real)}
    stack = [input_dir]
    while stack:
        current_dir = stack.pop()
        try:
            dir_entries = list(os.scandir(current_dir))
        except OSError as exc:
            logger.error(f"Cannot list directory {current_dir}: {exc}")
            failed.append({"input": current_dir, "error": f"cannot list directory: {exc}"})
            continue
        for entry in dir_entries:
            logical_path = entry.path
            try:
                is_dir = entry.is_dir(follow_symlinks=True)
                is_file = entry.is_file(follow_symlinks=True)
            except OSError as exc:
                logger.error(f"Cannot stat {logical_path}: {exc}")
                failed.append({"input": logical_path, "error": f"cannot stat: {exc}"})
                continue
            real_path = os.path.realpath(logical_path)
            if is_dir:
                if not _is_within(real_path, root_real):
                    logger.warning(f"Directory escapes input_dir, skipping: {logical_path} -> {real_path}")
                    failed.append(
                        {
                            "input": logical_path,
                            "error": f"directory symlink escapes input_dir: {real_path}",
                        }
                    )
                elif os.path.normcase(real_path) in visited:
                    logger.warning(f"Symlink cycle detected, skipping: {logical_path} -> {real_path}")
                    failed.append(
                        {
                            "input": logical_path,
                            "error": f"symlink cycle detected: {real_path}",
                        }
                    )
                else:
                    visited.add(os.path.normcase(real_path))
                    stack.append(logical_path)
            elif is_file:
                if not _is_within(real_path, root_real):
                    logger.warning(f"File escapes input_dir, skipping: {logical_path} -> {real_path}")
                    failed.append(
                        {
                            "input": logical_path,
                            "error": f"file symlink escapes input_dir: {real_path}",
                        }
                    )
                else:
                    entries.append((logical_path, real_path))
            # Anything else (sockets, dangling links, ...) is ignored
    return entries


def _output_path_for(logical_path: str, input_dir: str, output_root: str, extension: str) -> str:
    """Mirror the entry-relative logical path under output_root.

    The relative path is computed from the logical (entry) path, not the
    realpath, so symlinked directories keep their own subtree and same-named
    files from different subdirectories never collide.
    """
    relative = os.path.relpath(logical_path, input_dir)
    relative = os.path.splitext(relative)[0] + extension
    return os.path.normpath(os.path.join(output_root, relative))


def _render_thumbnail(real_path: str, target_size: tuple, keep_aspect_ratio: bool, pil_format: str, output_path: str):
    """Decode, resize and encode a single image. Raises on any failure."""
    with Image.open(real_path) as img:
        img.load()
        if keep_aspect_ratio:
            # thumbnail() only shrinks, never enlarges
            img.thumbnail(target_size)
        else:
            img = img.resize(target_size)

        if pil_format == "JPEG":
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            if has_alpha:
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, format=pil_format)
        return list(img.size)


def register_tool(mcp: FastMCP):
    @mcp.tool()
    def generate_thumbnails(
        input_dir: Annotated[str, Field(description="Full path to the directory containing source images (must be a full path)")],
        output_dir: Annotated[str, Field(description="Full path to the directory where thumbnails are written (must be a full path). The input directory layout is mirrored inside it.")],
        target_size: Annotated[
            Optional[list],
            Field(
                description=(
                    "Target thumbnail size as [width, height] in pixels. "
                    "With keep_aspect_ratio=True this is the maximum bounding box "
                    "(images are only shrunk, never enlarged); otherwise images "
                    "are stretched to exactly this size."
                )
            ),
        ] = None,
        output_format: Annotated[
            Optional[str],
            Field(
                description=(
                    "Output image format: 'png', 'jpeg' (or 'jpg'), 'webp', 'bmp', "
                    "'gif' or 'tiff'. Defaults to 'png'."
                )
            ),
        ] = None,
        keep_aspect_ratio: Annotated[
            bool,
            Field(
                description=(
                    "If True, preserve aspect ratio and only shrink images that "
                    "exceed target_size. If False, stretch every image to exactly "
                    "target_size."
                )
            ),
        ] = True,
        overwrite: Annotated[
            bool,
            Field(
                description=(
                    "If False (default), existing output files are reported as "
                    "skipped and the source image is not decoded or re-encoded."
                )
            ),
        ] = False,
    ) -> dict:
        """
        Generate thumbnails for every image in a directory tree in one batch.

        Recursively walks input_dir, resolves each entry with realpath and
        verifies with commonpath that it stays inside realpath(input_dir)
        (symlinks escaping the root or forming cycles are reported as failed).
        Output paths mirror the entry-relative directory layout under
        output_dir so files with the same name in different subdirectories
        never overwrite each other.

        Returns:
            A dict with per-file details grouped as 'processed', 'skipped'
            and 'failed', plus summary counts.
        """
        size = [256, 256] if target_size is None else list(target_size)
        if len(size) != 2 or any(not isinstance(dim, int) or dim <= 0 for dim in size):
            logger.error(f"Invalid target_size: {target_size}")
            raise ValueError("target_size must be [width, height] with positive integers")
        target = (size[0], size[1])

        format_name = (output_format or "png").lower()
        if format_name not in FORMAT_ALIASES:
            logger.error(f"Invalid output_format: {output_format}")
            raise ValueError(
                f"Invalid output_format. Choose from: {', '.join(sorted(FORMAT_ALIASES))}"
            )
        pil_format = FORMAT_ALIASES[format_name]
        extension = FORMAT_EXTENSIONS[pil_format]

        logger.info(
            f"generate_thumbnails requested: input_dir={input_dir}, output_dir={output_dir}, "
            f"target_size={target}, output_format={pil_format}, "
            f"keep_aspect_ratio={keep_aspect_ratio}, overwrite={overwrite}"
        )

        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(
                f"Input directory not found: {input_dir}. Please provide a full path to the directory."
            )

        root_real = os.path.realpath(input_dir)
        output_root = os.path.normpath(os.path.abspath(output_dir))
        os.makedirs(output_root, exist_ok=True)

        processed = []
        skipped = []
        failed = []

        # Batch traversal: resolve real paths and adjudicate containment once
        entries = _collect_files(input_dir, root_real, failed)
        logger.info(f"Found {len(entries)} candidate files under {input_dir}")

        for logical_path, real_path in entries:
            if os.path.splitext(logical_path)[1].lower() not in IMAGE_EXTENSIONS:
                logger.debug(f"Not an image, skipping: {logical_path}")
                skipped.append({"input": logical_path, "reason": "not a supported image file"})
                continue

            output_path = _output_path_for(logical_path, input_dir, output_root, extension)
            if not _is_within(output_path, output_root):
                logger.warning(f"Output path escapes output_dir, skipping: {output_path}")
                failed.append(
                    {
                        "input": logical_path,
                        "error": f"resolved output path escapes output_dir: {output_path}",
                    }
                )
                continue

            if not overwrite and os.path.exists(output_path):
                logger.debug(f"Output exists and overwrite=False, skipping: {output_path}")
                skipped.append({"input": logical_path, "output": output_path, "reason": "output already exists"})
                continue

            try:
                final_size = _render_thumbnail(real_path, target, keep_aspect_ratio, pil_format, output_path)
            except Exception as exc:
                # A single corrupt/undecodable file must not abort the batch
                logger.error(f"Failed to process {logical_path}: {exc}")
                failed.append({"input": logical_path, "error": str(exc)})
                continue

            logger.info(f"Thumbnail written: {output_path} ({final_size[0]}x{final_size[1]})")
            processed.append(
                {
                    "input": logical_path,
                    "output": output_path,
                    "size": final_size,
                }
            )

        result = {
            "input_dir": input_dir,
            "output_dir": output_root,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "summary": {
                "processed": len(processed),
                "skipped": len(skipped),
                "failed": len(failed),
            },
        }
        logger.info(f"generate_thumbnails finished: {result['summary']}")
        return result
