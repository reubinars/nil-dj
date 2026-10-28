import numpy as np
import pytest

from nil_dj.index import Final, LangRow
from nil_dj.languages import (
    NAMES,
    Overrides,
    Verdict,
    Window,
    decode_raw,
    encode_failed,
    encode_raw,
    load_overrides,
    normalise_code,
    parse_prompt,
    resolve,
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


@pytest.mark.parametrize(
    "text, langs, rest",
    [
        ("Japanese or English city pop", ["ja", "en"], "city pop"),
        ("japanese, korean and chinese ballads", ["ja", "ko", "zh"], "ballads"),
        ("instrumental for focus", ["instrumental"], "for focus"),
        ("songs with no vocals", ["instrumental"], "songs with"),
        ("lagu bahasa indonesia", ["id"], "lagu"),
        ("日本語 rock", ["ja"], "rock"),
        ("español", ["es"], "music"),
        ("J-pop for a rainy day", [], "J-pop for a rainy day"),
        ("French house bangers", [], "French house bangers"),
        ("Latin jazz", [], "Latin jazz"),
        ("Spanish guitar", [], "Spanish guitar"),
        ("K-pop and Korean ballads", ["ko"], "K-pop and ballads"),
        ("Englishman in New York", [], "Englishman in New York"),
        ("Latina summer", [], "Latina summer"),
        ("rock and roll", [], "rock and roll"),
    ],
)
def test_parse_prompt(text, langs, rest):
    assert parse_prompt(text) == (langs, rest)


# --- overrides and resolve ---

NO = Overrides({}, {}, {})


def raw_of(ns, **p):
    return encode_raw([Window(p, ns)])


def row(path, artist="A", album_artist=None, album="Al", raw=None):
    return LangRow(path, artist, album_artist, album, raw)


SUNG_EN = raw_of(0.1, en=0.95)
SUNG_ID = raw_of(0.1, id=0.95)
UNSURE_EN = raw_of(0.1, en=0.4, ja=0.3, ko=0.3)


def test_resolve_order():
    ov = Overrides(
        {"e:/m/t.flac": ("ko", None)},
        {("a", "al"): ("fr", None)},
        {"a": ("de", None)},
    )
    rows = [
        row("E:\\M\\T.flac", raw=SUNG_EN),  # track override
        row("x/b.flac", raw=SUNG_EN),  # album override
        row("x/c.flac", album="Other", raw=SUNG_EN),  # artist override
    ]
    f = resolve(rows, ov)
    assert [(f[r.path].language, f[r.path].source) for r in rows] == [
        ("ko", "override"),
        ("fr", "override"),
        ("de", "override"),
    ]
    # artist majority beats whisper, whisper is the fallback
    rows = [row(f"x/{i}.flac", raw=SUNG_ID) for i in range(3)] + [
        row("x/u.flac", raw=UNSURE_EN),
        row("y/z.flac", artist="B", raw=SUNG_EN),
    ]
    f = resolve(rows, NO)
    assert (f["x/u.flac"].language, f["x/u.flac"].source) == ("id", "artist")
    assert (f["y/z.flac"].language, f["y/z.flac"].source) == ("en", "whisper")
    assert f["y/z.flac"].conf == pytest.approx(0.95)
    assert f["y/z.flac"].vocal_score == pytest.approx(0.9)


def test_track_override_matches_path_key():
    ov = Overrides({"e:/music/a/x.flac": ("ja", None)}, {}, {})
    r = row("E:\\Music\\A\\x.flac", raw=SUNG_EN)
    assert resolve([r], ov)[r.path].language == "ja"
    r2 = row("e:/music/a/X.FLAC", raw=SUNG_EN)
    assert resolve([r2], ov)[r2.path].source == "override"


def test_override_keeps_vocal_score_and_has_no_conf():
    ov = Overrides({}, {}, {"a": ("ja", None)})
    r = row("p", raw=SUNG_EN)
    f = resolve([r], ov)["p"]
    assert f.conf is None and f.vocal_score == pytest.approx(0.9)
    failed = row("q", raw=encode_failed("boom"))
    g = resolve([failed], ov)["q"]
    assert (g.language, g.conf, g.vocal_score) == ("ja", None, None)


def test_album_override_uses_album_artist_else_artist():
    ov = Overrides({}, {("va", "hits"): ("es", None)}, {})
    with_aa = row("a", artist="Someone", album_artist="VA", album="Hits", raw=SUNG_EN)
    no_aa = row("b", artist="Va", album="Hits", raw=SUNG_EN)
    wrong = row("c", artist="Va", album_artist="Other", album="Hits", raw=SUNG_EN)
    f = resolve([with_aa, no_aa, wrong], ov)
    assert [f[p].language for p in "abc"] == ["es", "es", "en"]


def test_artist_override_matches_album_artist_or_artist():
    ov = Overrides({}, {}, {"band": ("ko", None)})
    by_artist = row("a", artist="Band", raw=SUNG_EN)
    by_aa = row("b", artist="Guest", album_artist="Band", raw=SUNG_EN)
    f = resolve([by_artist, by_aa], ov)
    assert f["a"].language == f["b"].language == "ko"


def test_names_compare_casefolded(tmp_path):
    p = tmp_path / "languages.toml"
    p.write_text('[artist]\n"BLACKPINK" = "ko"\n[album."ÉMILIE"]\n"Up" = "fr"\n')
    ov, problems = load_overrides(p)
    assert problems == []
    r = row("a", artist="Blackpink", raw=SUNG_EN)
    r2 = row("b", artist="x", album_artist="émilie", album="UP", raw=SUNG_EN)
    f = resolve([r, r2], ov)
    assert (f["a"].language, f["b"].language) == ("ko", "fr")


def _artist_rows(raws):
    return [row(f"t/{i}.flac", raw=r) for i, r in enumerate(raws)]


def test_artist_majority_thresholds():
    rows = _artist_rows([SUNG_ID] * 3 + [UNSURE_EN])
    f = resolve(rows, NO)
    assert (f["t/3.flac"].language, f["t/3.flac"].source) == ("id", "artist")
    assert f["t/3.flac"].conf == pytest.approx(0.4)
    f = resolve(_artist_rows([SUNG_ID] * 2 + [UNSURE_EN]), NO)
    assert (f["t/2.flac"].language, f["t/2.flac"].source) == ("en", "whisper")
    f = resolve(_artist_rows([SUNG_ID] * 3 + [SUNG_EN] * 2 + [UNSURE_EN]), NO)  # 60 %
    assert f["t/5.flac"].source == "whisper"
    f = resolve(_artist_rows([SUNG_ID] * 7 + [SUNG_EN] * 3 + [UNSURE_EN]), NO)  # 70 %
    assert f["t/10.flac"].language == "id"


def test_artist_majority_ignores_overridden_failed_and_other_artists():
    ov = Overrides({"t/0.flac": ("en", None)}, {}, {})
    rows = _artist_rows([SUNG_ID] * 3 + [encode_failed("x")] + [UNSURE_EN])
    f = resolve(rows, ov)
    assert f["t/4.flac"].source == "whisper"  # the overridden track no longer counts
    other = [row(f"o/{i}.flac", artist="B", raw=SUNG_ID) for i in range(3)]
    f = resolve(other + [row("t/u.flac", raw=UNSURE_EN)], NO)
    assert f["t/u.flac"].source == "whisper"


def test_artist_majority_groups_by_album_artist():
    rows = [row(f"t/{i}.flac", artist=f"g{i}", album_artist="Band", raw=SUNG_ID) for i in range(3)]
    rows.append(row("t/u.flac", artist="g9", album_artist="Band", raw=UNSURE_EN))
    assert resolve(rows, NO)["t/u.flac"].source == "artist"


def test_artist_majority_never_touches_instrumentals():
    instr = raw_of(0.9, ja=0.4, en=0.3, ko=0.3)
    rows = _artist_rows([SUNG_ID] * 3 + [instr])
    f = resolve(rows, NO)
    assert (f["t/3.flac"].language, f["t/3.flac"].source) == ("instrumental", "whisper")
    # and instrumentals do not vote
    f = resolve(_artist_rows([instr] * 3 + [UNSURE_EN]), NO)
    assert f["t/3.flac"].source == "whisper"


def test_failed_raw_gives_unknown():
    r = row("a", raw=encode_failed("decode error"))
    assert resolve([r], NO)["a"] == Final("unknown", None, None, None, "whisper")


def test_rows_without_raw_are_skipped():
    assert resolve([row("a", raw=None)], NO) == {}


def test_mixed_override_value(tmp_path):
    p = tmp_path / "languages.toml"
    p.write_text("[track]\n'E:\\Music\\x.flac' = \"JA+en\"\n[artist]\nQ = 'instrumental'\n")
    ov, problems = load_overrides(p)
    assert problems == []
    assert ov.tracks == {"e:/music/x.flac": ("ja", "en")}
    assert ov.artists == {"q": ("instrumental", None)}
    r = row("E:/Music/X.flac", raw=SUNG_EN)
    f = resolve([r], ov)[r.path]
    assert (f.language, f.alt) == ("ja", "en")


def test_missing_override_file(tmp_path):
    ov, problems = load_overrides(tmp_path / "nope.toml")
    assert ov == NO and problems == []


@pytest.mark.parametrize(
    "toml, problem",
    [
        ("[artist]\nX = 'xx'", "unknown language 'xx'"),
        ("[artist]\nX = 3", "must be a language code"),
        ("[album]\nX = 'ja'", "[album] needs an artist table"),
        ("[bogus]\n", "unknown section"),
        ("[artist\n", "languages.toml:"),
    ],
)
def test_bad_override_file_is_ignored_whole(tmp_path, toml, problem):
    p = tmp_path / "languages.toml"
    p.write_text(toml)
    ov, problems = load_overrides(p)
    assert ov == NO
    assert len(problems) >= 1
    assert all(m.startswith("languages.toml:") for m in problems)
    assert any(problem in m for m in problems)


def test_problem_message_names_section_and_key(tmp_path):
    p = tmp_path / "languages.toml"
    p.write_text("[artist]\nX = 'xx'\n")
    assert load_overrides(p)[1] == ['languages.toml: [artist] "X": unknown language \'xx\'']


def test_resolve_index_writes_finals_and_meta(tmp_path):
    from dataclasses import replace

    from nil_dj.index import Index
    from nil_dj.languages import overrides_changed, resolve_index
    from nil_dj.tags import TrackInfo

    base = TrackInfo(
        title="t", artist="A", album_artist=None, album="al", genre=None, year=2001, disc=1,
        number=1, duration_s=200.0, codec="flac", sample_rate=44100, bits=16,
    )  # fmt: skip
    idx = Index.open(tmp_path / "i.sqlite", "m")
    idx.upsert("p", 1, 1, replace(base), 120.0, -9.0, np.zeros(512, np.float32))
    idx.set_language_raw("p", SUNG_EN)
    ov = tmp_path / "languages.toml"
    try:
        assert overrides_changed(idx, ov)
        assert resolve_index(idx, ov) == []
        assert idx.get_meta("overrides_mtime_ns") == "0"
        assert not overrides_changed(idx, ov)
        assert idx.language_rows()[0].raw == SUNG_EN
        ov.write_text("[artist]\nA = 'ko'\n")
        assert overrides_changed(idx, ov)
        assert resolve_index(idx, ov) == []
        assert not overrides_changed(idx, ov)
        ov.write_text("[bogus]\n")
        assert resolve_index(idx, ov) != []
    finally:
        idx.close()
