import os

import pytest
from fastmcp import Client, FastMCP
from PIL import Image

from imagesorcery_mcp.server import mcp as image_sorcery_mcp_server


@pytest.fixture
def mcp_server():
    # Use the existing server instance
    return image_sorcery_mcp_server


@pytest.fixture
def nested_input(tmp_path):
    """Create a nested directory tree of source images."""
    input_dir = tmp_path / "assets"
    (input_dir / "icons").mkdir(parents=True)
    (input_dir / "photos" / "2026").mkdir(parents=True)

    # Root-level image, larger than the default target
    Image.new("RGB", (800, 400), color=(200, 30, 30)).save(input_dir / "hero.png")
    # Same filename in two different subdirectories
    Image.new("RGB", (300, 300), color=(30, 200, 30)).save(input_dir / "icons" / "cover.png")
    Image.new("RGB", (500, 250), color=(30, 30, 200)).save(input_dir / "photos" / "2026" / "cover.png")
    # An image smaller than the target size
    Image.new("RGB", (64, 32), color=(200, 200, 30)).save(input_dir / "icons" / "tiny.png")
    # An RGBA image (alpha channel)
    Image.new("RGBA", (400, 200), color=(255, 0, 0, 128)).save(input_dir / "photos" / "alpha.png")
    # A non-image file
    (input_dir / "notes.txt").write_text("not an image")
    # A corrupt file with an image extension
    (input_dir / "broken.png").write_bytes(b"this is not a real png")

    return input_dir


async def _run(mcp_server, input_dir, output_dir, **kwargs):
    args = {"input_dir": str(input_dir), "output_dir": str(output_dir)}
    args.update(kwargs)
    async with Client(mcp_server) as client:
        result = await client.call_tool("generate_thumbnails", args)
    return result.data


class TestGenerateThumbnailsToolDefinition:
    """Tests for the generate_thumbnails tool definition and metadata."""

    @pytest.mark.asyncio
    async def test_generate_thumbnails_in_tools_list(self, mcp_server: FastMCP):
        """Tests that generate_thumbnails is in the list of available tools."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert tools, "Tools list should not be empty"
            tool_names = [tool.name for tool in tools]
            assert "generate_thumbnails" in tool_names

    @pytest.mark.asyncio
    async def test_generate_thumbnails_parameters(self, mcp_server: FastMCP):
        """Tests that generate_thumbnails exposes the expected parameters."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool = next(t for t in tools if t.name == "generate_thumbnails")

        properties = tool.inputSchema["properties"]
        for param in [
            "input_dir",
            "output_dir",
            "target_size",
            "output_format",
            "keep_aspect_ratio",
            "overwrite",
        ]:
            assert param in properties, f"missing parameter: {param}"


class TestGenerateThumbnailsExecution:
    """Tests for the generate_thumbnails batch execution."""

    @pytest.mark.asyncio
    async def test_nested_batch_and_layout_mirroring(self, mcp_server, nested_input, tmp_path):
        """All images in the tree are processed and the layout is mirrored."""
        output_dir = tmp_path / "thumbs"
        data = await _run(
            mcp_server, nested_input, output_dir, target_size=[128, 128]
        )

        assert data["summary"]["processed"] == 5
        assert data["summary"]["skipped"] == 1  # notes.txt
        assert data["summary"]["failed"] == 1  # broken.png

        # Same-named cover.png from different subdirs must not collide
        cover_icons = output_dir / "icons" / "cover.png"
        cover_photos = output_dir / "photos" / "2026" / "cover.png"
        assert cover_icons.exists()
        assert cover_photos.exists()
        assert Image.open(cover_icons).size == (128, 128)
        assert Image.open(cover_photos).size == (128, 64)  # 500x250 keeps ratio

        # Aspect ratio preserved, only shrunk
        assert Image.open(output_dir / "hero.png").size == (128, 64)

        # Failure detail carries the reason
        failed_inputs = [os.path.basename(f["input"]) for f in data["failed"]]
        assert "broken.png" in failed_inputs
        assert data["failed"][0]["error"]

    @pytest.mark.asyncio
    async def test_keep_aspect_ratio_never_upscales(self, mcp_server, nested_input, tmp_path):
        """Images smaller than target_size keep their original dimensions."""
        output_dir = tmp_path / "thumbs"
        await _run(mcp_server, nested_input, output_dir, target_size=[128, 128])
        assert Image.open(output_dir / "icons" / "tiny.png").size == (64, 32)

    @pytest.mark.asyncio
    async def test_stretch_without_aspect_ratio(self, mcp_server, nested_input, tmp_path):
        """keep_aspect_ratio=False stretches to the exact target size."""
        output_dir = tmp_path / "thumbs"
        await _run(
            mcp_server,
            nested_input,
            output_dir,
            target_size=[100, 100],
            keep_aspect_ratio=False,
        )
        assert Image.open(output_dir / "hero.png").size == (100, 100)
        assert Image.open(output_dir / "icons" / "tiny.png").size == (100, 100)

    @pytest.mark.asyncio
    async def test_jpeg_flattens_alpha_on_white(self, mcp_server, nested_input, tmp_path):
        """JPEG output composites alpha images on a white background."""
        output_dir = tmp_path / "thumbs"
        data = await _run(
            mcp_server, nested_input, output_dir, output_format="jpeg"
        )

        assert data["summary"]["failed"] == 1  # only broken.png
        out = output_dir / "photos" / "alpha.jpg"
        assert out.exists()
        with Image.open(out) as img:
            assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_overwrite_false_skips_existing(self, mcp_server, nested_input, tmp_path):
        """Existing outputs are skipped without decoding or re-encoding."""
        output_dir = tmp_path / "thumbs"
        first = await _run(mcp_server, nested_input, output_dir, target_size=[128, 128])
        assert first["summary"]["processed"] == 5

        hero_out = output_dir / "hero.png"
        original_bytes = hero_out.read_bytes()

        second = await _run(mcp_server, nested_input, output_dir, target_size=[64, 64])
        assert second["summary"]["processed"] == 0
        assert second["summary"]["skipped"] == 6  # 5 existing outputs + notes.txt

        # File untouched: still the 128px version, byte-identical
        assert hero_out.read_bytes() == original_bytes
        assert Image.open(hero_out).size == (128, 64)

    @pytest.mark.asyncio
    async def test_overwrite_true_regenerates(self, mcp_server, nested_input, tmp_path):
        """overwrite=True re-encodes existing outputs."""
        output_dir = tmp_path / "thumbs"
        await _run(mcp_server, nested_input, output_dir, target_size=[128, 128])
        data = await _run(
            mcp_server,
            nested_input,
            output_dir,
            target_size=[64, 64],
            overwrite=True,
        )
        assert data["summary"]["processed"] == 5
        assert Image.open(output_dir / "hero.png").size == (64, 32)

    @pytest.mark.asyncio
    async def test_missing_input_dir_raises(self, mcp_server, tmp_path):
        """A nonexistent input directory is a hard error."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            await _run(mcp_server, tmp_path / "nope", tmp_path / "out")


class TestGenerateThumbnailsSymlinks:
    """Symlink containment and cycle handling (skipped without privileges)."""

    @pytest.fixture
    def symlink_or_skip(self, tmp_path):
        """Create a symlink, skipping the test when the OS refuses."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks not available: {exc}")
        return link

    @pytest.mark.asyncio
    async def test_escaping_symlink_marked_failed(
        self, mcp_server, nested_input, tmp_path, symlink_or_skip
    ):
        """A symlink pointing outside input_dir is reported as failed."""
        outside = tmp_path / "outside"
        outside.mkdir()
        Image.new("RGB", (100, 100)).save(outside / "secret.png")

        link = nested_input / "escape_link"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks not available: {exc}")

        output_dir = tmp_path / "thumbs"
        data = await _run(mcp_server, nested_input, output_dir)

        failed_inputs = [os.path.basename(f["input"]) for f in data["failed"]]
        assert "escape_link" in failed_inputs
        assert not (output_dir / "escape_link").exists()
        # The rest of the batch still completed
        assert data["summary"]["processed"] == 5

    @pytest.mark.asyncio
    async def test_symlink_cycle_marked_failed(
        self, mcp_server, nested_input, tmp_path, symlink_or_skip
    ):
        """A symlink loop back into the tree is reported as failed, not traversed forever."""
        link = nested_input / "icons" / "loop"
        try:
            os.symlink(nested_input, link, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks not available: {exc}")

        output_dir = tmp_path / "thumbs"
        data = await _run(mcp_server, nested_input, output_dir)

        failed_inputs = [os.path.basename(f["input"]) for f in data["failed"]]
        assert "loop" in failed_inputs
        assert data["summary"]["processed"] == 5
