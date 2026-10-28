import pytest

from nil_dj.languages import (
    NAMES,
    Verdict,
    Window,
    decode_raw,
    encode_failed,
    encode_raw,
    normalise_code,
    verdict,
)


def w(ns, **p):
    return Window(p, ns)


def test_group_is_merged_before_anything():
    v = verdict([w(0.2, jw=0.5, ms=0.2, en=0.3)])
    assert (v.language, round(v.conf, 3)) == ("id", 0.7)


def test_instrumental_needs_low_conf_and_high_no_speech():
    assert verdict([w(0.8, ja=0.4, en=0.3, ko=0.3)]).language == "instrumental"
    assert verdict([w(0.8, ja=0.7, en=0.3)]).language == "ja"  # confident: sung
    v = verdict([w(0.5, ja=0.4, en=0.3, ko=0.3)])  # sung but unsure
    assert (v.language, v.unsure) == ("ja", True)


def test_weights_favour_vocal_windows():
    v = verdict([w(0.99, en=1.0), w(0.1, ja=1.0), w(0.1, ja=1.0)])
    assert v.language == "ja" and v.conf > 0.99
    assert round(v.vocal_score, 3) == round(1 - 1.19 / 3, 3)


def test_no_speech_one_still_counts_with_min_weight():
    assert verdict([w(1.0, fr=1.0)]).language == "fr"  # conf 1.0: sung, however quiet


def test_mixed_names_alt_from_confident_windows():
    v = verdict([w(0.2, ja=0.9, en=0.1), w(0.2, en=0.8, ja=0.2), w(0.2, ja=0.9, en=0.1)])
    assert (v.language, v.alt) == ("ja", "en")
    assert verdict([w(0.2, ja=0.9, en=0.1), w(0.2, en=0.55, ja=0.45)]).alt is None


def test_single_window():
    assert verdict([w(0.1, es=0.95)]) == Verdict("es", None, 0.95, 0.9, False)


def test_raw_round_trip_rounds_and_drops_tiny():
    text = encode_raw([w(0.123456, ja=0.987654, en=0.0004)])
    assert decode_raw(text) == [w(0.1235, ja=0.9877)]
    assert decode_raw(encode_failed("decode error")) is None


def test_normalise_code():
    assert normalise_code(" JW ") == "id" and normalise_code("instrumental") == "instrumental"
    with pytest.raises(ValueError, match="unknown language 'xx'"):
        normalise_code("xx")


def test_names_has_no_group_aliases():
    assert len(NAMES) == 96 and NAMES["id"] == "Indonesian"
    assert not {"jw", "ms", "su"} & set(NAMES)
