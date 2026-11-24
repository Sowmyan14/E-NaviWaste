"""Reading images from files: JPEG and PNG as before, HEIC/HEIF through the optional plugin."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import make_image
from PIL import Image

import electronics_classifier.cli as cli
from electronics_classifier.cli import HEIC_HINT, InputError, load_image, main

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_batch.py"
spec = importlib.util.spec_from_file_location("run_batch_loading", SCRIPT)
run_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_batch)

def need_heif():
    return pytest.importorskip("pillow_heif", reason=f"needs: {HEIC_HINT}")


def save_heic(path, array, **options):
    need_heif().register_heif_opener()
    Image.fromarray(array).save(path, format="HEIF", **options)
    return path


# --- the formats that already worked keep working -----------------------------------------


def test_png_is_read_losslessly_as_rgb_uint8(tmp_path):
    original = make_image("noise", 60, 90)
    path = tmp_path / "photo.png"
    Image.fromarray(original).save(path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8 and loaded.shape == (60, 90, 3)
    assert (loaded == original).all()  # PNG is lossless: every pixel identical
    assert loaded.flags.writeable


def test_jpeg_is_read_as_rgb_uint8_close_to_the_original(tmp_path):
    original = make_image("gradient", 60, 90)
    path = tmp_path / "photo.jpg"
    Image.fromarray(original).save(path, quality=95)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8 and loaded.shape == (60, 90, 3)
    assert np.abs(loaded.astype(int) - original.astype(int)).mean() < 3  # JPEG is lossy


@pytest.mark.parametrize("suffix", [".JPG", ".jpeg", ".PNG"])
def test_the_file_suffix_case_does_not_matter(tmp_path, suffix):
    path = tmp_path / f"photo{suffix}"
    fmt = "PNG" if "P" in suffix.upper().replace("JPG", "") else "JPEG"
    Image.fromarray(make_image("noise", 20, 30)).save(path, format=fmt)
    assert load_image(path).shape == (20, 30, 3)


def test_jpeg_and_png_never_need_the_heic_plugin(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pillow_heif", None)  # as if it were not installed
    monkeypatch.setattr(cli, "_heif_ready", None)
    for name, fmt in (("a.png", "PNG"), ("a.jpg", "JPEG")):
        Image.fromarray(make_image("noise", 16, 16)).save(tmp_path / name, format=fmt)
        assert load_image(tmp_path / name).shape == (16, 16, 3)
    assert cli._heif_ready is None  # the plugin was never even tried


# --- HEIC ---------------------------------------------------------------------------------


def test_heic_is_read_as_rgb_uint8_close_to_the_original(tmp_path):
    original = make_image("gradient", 64, 96)
    loaded = load_image(save_heic(tmp_path / "photo.heic", original, quality=95))
    assert loaded.dtype == np.uint8 and loaded.shape == (64, 96, 3)
    assert np.abs(loaded.astype(int) - original.astype(int)).mean() < 6  # HEIC is lossy


def test_heic_with_an_exif_rotation_comes_out_upright(tmp_path):
    need_heif()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, to be rotated 90 degrees clockwise
    sideways = make_image("gradient", 40, 80)
    path = save_heic(tmp_path / "sideways.heic", sideways, exif=exif.tobytes())
    assert load_image(path).shape == (80, 40, 3)  # width and height swapped


@pytest.mark.parametrize("name", ["photo.HEIC", "photo.heif", "photo.Heic"])
def test_heic_suffix_variants_are_read(tmp_path, name):
    assert load_image(save_heic(tmp_path / name, make_image("noise", 32, 32))).shape == (32, 32, 3)


def test_a_heic_file_with_the_wrong_name_is_still_recognised(tmp_path):
    path = save_heic(tmp_path / "actually_heic.jpg", make_image("noise", 32, 48))
    assert load_image(path).shape == (32, 48, 3)


def test_the_cli_runs_on_a_heic_photo(tmp_path, monkeypatch):
    monkeypatch.setenv("EC_USE_FAKES", "1")
    path = save_heic(tmp_path / "photo.heic", make_image("gradient", 300, 400))
    out = tmp_path / "out"
    assert main(["run", str(path), "--out-dir", str(out)]) == 0
    assert (out / "result.json").is_file()
    assert '"image_width": 400' in (out / "result.json").read_text()


def test_the_batch_runner_lists_heic_and_other_photos_but_not_other_files(tmp_path):
    for name in ("a.heic", "b.HEIC", "c.heif", "d.jpg", "e.png", "g.avif", "notes.txt", "f.pdf"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in run_batch.find_images(tmp_path)] == [
        "a.heic", "b.HEIC", "c.heif", "d.jpg", "e.png", "g.avif",
    ]  # fmt: skip


# --- clear errors --------------------------------------------------------------------------


def test_a_heic_file_without_the_plugin_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pillow_heif", None)  # as if it were not installed
    monkeypatch.setattr(cli, "_heif_ready", None)
    path = tmp_path / "iphone.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)  # not decodable here anyway
    with pytest.raises(InputError) as err:
        load_image(path)
    message = str(err.value)
    assert "HEIC support is not installed" in message and "electronics-classifier[heic]" in message
    assert "\n" not in message


def test_a_corrupt_heic_with_the_plugin_installed_reports_a_plain_read_error(tmp_path):
    need_heif()
    path = tmp_path / "broken.heic"
    path.write_bytes(b"this is not an image")
    with pytest.raises(InputError, match="cannot read image") as err:
        load_image(path)
    assert "not installed" not in str(err.value)  # the plugin is there; the file is just bad


def test_other_unreadable_files_keep_the_old_message_without_a_heic_hint(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(InputError, match="cannot read image") as err:
        load_image(path)
    assert "heic" not in str(err.value).lower()


def test_the_cli_reports_a_missing_heic_plugin_in_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EC_USE_FAKES", "1")
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    monkeypatch.setattr(cli, "_heif_ready", None)
    path = tmp_path / "iphone.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    assert main(["run", str(path), "--out-dir", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: cannot read HEIC image") and err.count("\n") == 1
    assert "electronics-classifier[heic]" in err


# --- what Android cameras produce -----------------------------------------------------------


def _scene():
    ramp = np.linspace(0, 255, 160, dtype=np.uint8)[None, :].repeat(120, axis=0)
    return np.stack([ramp, ramp[::-1], ramp // 2], axis=-1)  # 120 high, 160 wide


def _jpeg_with_orientation(tag):
    def make(path):
        exif = Image.Exif()
        exif[0x0112] = tag
        Image.fromarray(_scene()).save(path, exif=exif.tobytes())

    return make


def _motion_photo(path):
    """A Motion Photo is a normal JPEG with an MP4 video appended after it."""
    import io

    buffer = io.BytesIO()
    Image.fromarray(_scene()).save(buffer, format="JPEG")
    path.write_bytes(buffer.getvalue() + b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 20)


def _multi_picture(path):
    """Ultra HDR and portrait-depth photos are multi-picture (MPF) JPEGs."""
    scene = Image.fromarray(_scene())
    scene.save(path, format="MPO", save_all=True, append_images=[scene.resize((80, 60))])


ANDROID_FILES = {
    "camera.jpg": (lambda p: Image.fromarray(_scene()).save(p, quality=92), (120, 160)),
    "upright_exif3.jpg": (_jpeg_with_orientation(3), (120, 160)),  # rotated 180 degrees
    "portrait_exif6.jpg": (_jpeg_with_orientation(6), (160, 120)),  # phone held upright
    "portrait_exif8.jpg": (_jpeg_with_orientation(8), (160, 120)),
    "MVIMG_motion_photo.jpg": (_motion_photo, (120, 160)),
    "ultra_hdr_multi_picture.jpg": (_multi_picture, (120, 160)),
    "photo.webp": (lambda p: Image.fromarray(_scene()).save(p, format="WEBP"), (120, 160)),
    "photo.avif": (lambda p: Image.fromarray(_scene()).save(p, format="AVIF"), (120, 160)),
    "photo.heic": (lambda p: save_heic(p, _scene()), (120, 160)),
}


@pytest.mark.parametrize("name", ANDROID_FILES)
def test_android_camera_files_are_read_upright_as_rgb_uint8(tmp_path, name):
    make, expected_shape = ANDROID_FILES[name]
    path = tmp_path / name
    make(path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8 and loaded.shape == (*expected_shape, 3)


def test_the_picture_not_the_video_is_what_a_motion_photo_gives(tmp_path):
    path = tmp_path / "MVIMG_20260101.jpg"
    _motion_photo(path)
    original = _scene().astype(int)
    assert np.abs(load_image(path).astype(int) - original).mean() < 4  # the still, undamaged


def test_a_raw_dng_gets_a_clear_one_line_message_instead_of_a_crash(tmp_path, capsys):
    path = tmp_path / "pro_mode.dng"
    path.write_bytes(b"II*\x00" + b"\x08\x00\x00\x00" + bytes(64))  # TIFF-style header, no pixels
    with pytest.raises(InputError, match="RAW/DNG photos are not supported") as err:
        load_image(path)
    assert "\n" not in str(err.value) and "JPEG or PNG" in str(err.value)
    assert main(["run", str(path), "--out-dir", str(tmp_path / "o")]) == 2
    assert capsys.readouterr().err.startswith("error: cannot read RAW image")
