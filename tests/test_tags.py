import struct

import pytest
from mutagen.id3 import TDRC, TIT2, TPE1, TPOS, TRCK
from mutagen.wave import WAVE

from nil_dj.tags import TagError, read_info

DECO = {
    "title": "ネオネオン",
    "artist": "DECO*27",
    "albumartist": "DECO*27",
    "album": "アンデッドアリス",
    "genre": "J-Pop",
    "date": "2020-05-01",
    "discnumber": "1/2",
    "tracknumber": "3/12",
}


def test_flac_tags_and_info(make_audio):
    p = make_audio("n.flac", 3, rate=96000, bits=24, tags=DECO)
    i = read_info(p)
    assert (i.title, i.artist, i.album_artist, i.album, i.genre) == (
        "ネオネオン", "DECO*27", "DECO*27", "アンデッドアリス", "J-Pop",
    )  # fmt: skip
    assert (i.year, i.disc, i.number) == (2020, 1, 3)
    assert (i.codec, i.bits, i.sample_rate) == ("flac", 24, 96000)
    assert abs(i.duration_s - 3) <= 0.05


def test_m4a_aac_and_alac(make_audio):
    aac = read_info(make_audio("a.m4a", 2, codec="aac"))
    assert (aac.codec, aac.bits) == ("aac", None)
    alac = read_info(make_audio("b.m4a", 2, codec="alac"))
    assert (alac.codec, alac.bits) == ("alac", 16)


def test_m4a_tags(make_audio):
    i = read_info(make_audio("t.m4a", 2, codec="alac", tags=DECO))
    assert (i.title, i.album_artist, i.year, i.disc, i.number) == ("ネオネオン", "DECO*27", 2020, 1, 3)


def test_wav_and_aiff(make_audio):
    assert read_info(make_audio("a.wav", 2, codec="wav")).codec == "wav"
    aiff = read_info(make_audio("a.aiff", 2, codec="aiff", bits=24))
    assert (aiff.codec, aiff.bits) == ("aiff", 24)


def test_wav_id3_tags(make_audio):
    p = make_audio("t.wav", 2, codec="wav")
    f = WAVE(p)
    f.add_tags()
    for frame in (TIT2(text="雨"), TPE1(text="Someone"), TDRC(text="1999"), TPOS(text="2"),
                  TRCK(text="7/9")):  # fmt: skip
        f.tags.add(frame)
    f.save()
    i = read_info(p)
    assert (i.title, i.artist, i.year, i.disc, i.number) == ("雨", "Someone", 1999, 2, 7)


def _write_dsf(path, seconds=1, fs=2822400):
    samples = fs * seconds
    block = 4096
    per_channel = -(-samples // 8 // block) * block
    data = bytes([0x69]) * (per_channel * 2)
    fmt = struct.pack("<4sQIIIIIIQII", b"fmt ", 52, 1, 0, 2, 2, fs, 1, samples, block, 0)
    body = fmt + struct.pack("<4sQ", b"data", 12 + len(data)) + data
    path.write_bytes(struct.pack("<4sQQQ", b"DSD ", 28, 28 + len(body), 0) + body)
    return path


def test_dsf_is_dsd(tmp_path):
    i = read_info(_write_dsf(tmp_path / "d.dsf"))
    assert (i.codec, i.bits, i.sample_rate) == ("dsd", None, 2822400)
    assert abs(i.duration_s - 1) <= 0.05


def test_missing_tags_are_none(make_audio):
    i = read_info(make_audio("plain.flac", 2))
    assert (i.title, i.artist, i.album_artist, i.album, i.genre) == (None,) * 5
    assert (i.year, i.disc, i.number) == (None, None, None)
    assert i.codec == "flac" and i.bits == 16


def test_garbage_raises_tag_error(tmp_path):
    p = tmp_path / "x.flac"
    p.write_bytes(b"this is a text file")
    with pytest.raises(TagError):
        read_info(p)
    empty = tmp_path / "e.flac"
    empty.write_bytes(b"")
    with pytest.raises(TagError):
        read_info(empty)
