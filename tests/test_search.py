from dataclasses import replace

import pytest
from fakes import BASE, ChosenText, unit

from nil_dj.index import Final, Index
from nil_dj.search import (
    Filters,
    candidate_pool_size,
    describe_format,
    filters_from,
    find_by_tags,
    rank,
    search_text,
    similar_to_album,
    similar_to_paths,
)


@pytest.fixture
def idx(tmp_path):
    i = Index.open(tmp_path / "i.sqlite", "fake")
    yield i
    i.close()


def add(idx, path, v, **kw):
    idx.upsert(path, 1, 1, replace(BASE, **kw), None, None, v)


def paths(hits):
    return [h.row.path for h in hits]


def test_rank_by_cosine_with_path_tiebreak(idx):
    add(idx, "c", unit(1, 0))
    add(idx, "b", unit(1, 1))
    add(idx, "a", unit(1, 1))  # same score as b: path order decides
    add(idx, "d", unit(0, 1))
    hits = search_text(idx, ChosenText(unit(1, 0)), "q")
    assert paths(hits) == ["c", "a", "b", "d"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(hits[2].score)
    assert paths(search_text(idx, ChosenText(unit(1, 0)), "q", limit=2)) == ["c", "a"]


def test_filters_each(idx):
    add(idx, "short", unit(1), duration_s=60)
    add(idx, "long", unit(1), duration_s=600)
    add(idx, "old", unit(1), year=1970)
    add(idx, "nyear", unit(1), year=None)
    add(idx, "aa", unit(1), artist="Feat", album_artist="Björk")
    add(idx, "dsd", unit(1), codec="dsd", sample_rate=2822400, bits=None)
    add(idx, "hires", unit(1), bits=24, sample_rate=96000)
    add(idx, "aac", unit(1), codec="aac", bits=None)
    q = ChosenText(unit(1))

    def run(**kw):
        return set(paths(search_text(idx, q, "q", limit=100, filters=Filters(**kw))))

    everything = run()
    assert len(everything) == 8
    assert "short" not in run(min_minutes=2)
    assert "long" not in run(max_minutes=5)
    assert run(year_from=1990) == everything - {"old", "nyear"}
    assert run(year_to=1990) == {"old"}
    assert run(artists=["björk"]) == {"aa"}  # album artist, case-insensitive beyond ASCII
    assert run(artists=["FEAT"]) == {"aa"}  # or the track artist
    assert run(exclude_artists=["a"]) == {"aa"}
    assert run(formats=["dsd", "aac"]) == {"dsd", "aac"}
    assert run(hires_only=True) == {"dsd", "hires"}
    with pytest.raises(ValueError, match="mp3"):
        Filters(formats=["mp3"]).to_sql()


def test_template_applied(idx):
    q = ChosenText(unit(1))
    search_text(idx, q, "rain", template="This is a sound of {}")
    assert q.seen == ["This is a sound of rain"]


def test_similar_excludes_seeds(idx):
    add(idx, "s1", unit(1, 0))
    add(idx, "s2", unit(0, 1))
    add(idx, "mid", unit(1, 1))
    add(idx, "far", unit(-1, 0.1))
    assert paths(similar_to_paths(idx, ["s1", "s2"])) == ["mid", "far"]


def test_similar_respects_filters(idx):
    add(idx, "seed", unit(1), year=1970)
    add(idx, "x", unit(1), year=1970)
    add(idx, "y", unit(1), year=2010)
    assert paths(similar_to_paths(idx, ["seed"], filters=Filters(year_from=2000))) == ["y"]


def test_similar_album_mean(idx):
    add(idx, "al1", unit(1, 0), album="Blue", artist="Joni")
    add(idx, "al2", unit(0, 1), album="Blue", artist="Joni")
    add(idx, "other_blue", unit(-1, -1), album="Blue", artist="Someone")
    add(idx, "mid", unit(1, 1), album="X")
    add(idx, "near1", unit(1, 0.1), album="X")
    hits = similar_to_album(idx, "blue", "JONI")
    assert paths(hits)[:2] == ["mid", "near1"]
    assert "al1" not in paths(hits) and "al2" not in paths(hits)
    assert len(similar_to_album(idx, "BLUE", None)) == 2  # all three Blue tracks are seeds


def test_unknown_seed_keyerror(idx):
    add(idx, "a", unit(1))
    with pytest.raises(KeyError, match="missing"):
        similar_to_paths(idx, ["a", "missing"])
    with pytest.raises(KeyError):
        similar_to_album(idx, "No Such Album", None)


def test_empty_results(idx):
    assert search_text(idx, ChosenText(unit(1)), "q") == []
    add(idx, "a", unit(1), year=2000)
    assert search_text(idx, ChosenText(unit(1)), "q", filters=Filters(year_from=2100)) == []


def test_candidate_pool_size():
    assert candidate_pool_size(5) == 60
    assert candidate_pool_size(30) == 90


def test_describe_format():
    def fmt(**kw):
        return describe_format(replace(BASE, **kw))

    assert fmt(bits=24, sample_rate=96000) == "flac 24-bit/96 kHz"
    assert fmt() == "flac 16-bit/44.1 kHz"
    assert fmt(codec="aac", bits=None) == "aac 44.1 kHz"
    assert fmt(codec="dsd", bits=None, sample_rate=2822400) == "dsd 2.8 MHz"
    assert fmt(codec="dsd", bits=None, sample_rate=5644800) == "dsd 5.6 MHz"


def test_find_by_tags(idx):
    add(idx, "z", unit(1), artist="Feat", album_artist="Björk", album="Homogenic", number=2)
    add(idx, "y", unit(1), artist="Björk", album="Homogenic", number=1)
    add(idx, "x", unit(1), artist="Joni", album="Blue", title="River")
    add(idx, "w", unit(1), artist="Joni", album="Blue", title="Carey", disc=None)

    def found(**kw):
        return [r.path for r in find_by_tags(idx, **kw)]

    assert found(artist="BJÖR") == ["y", "z"]  # album_artist counts; album order
    assert found(album="geni") == ["y", "z"]
    assert found(title="riv") == ["x"]
    assert found(artist="joni", title="car") == ["w"]
    assert found(artist="nobody") == []
    assert found(artist="j", limit=1) == ["y"]  # ordered by album artist: björk < joni


def test_rank_returns_vectors_and_excludes(idx):
    add(idx, "a", unit(1, 0))
    add(idx, "b", unit(0, 1), year=1970)
    hits = rank(idx, unit(1, 0), 10)
    assert [h.row.path for h in hits] == ["a", "b"]
    assert hits[0].vector == pytest.approx(unit(1, 0))
    assert paths(rank(idx, unit(1, 0), 10, exclude=frozenset({"a"}))) == ["b"]
    assert paths(rank(idx, unit(1, 0), 10, Filters(year_to=1980))) == ["b"]


def label(idx, **finals):
    idx.set_final({p: Final(*v, None, None, "override") for p, v in finals.items()})


def langs_idx(idx):
    for p in "abcde":
        add(idx, p, unit(1, 0))
    label(idx, a=("ja", None), b=("en", "ja"), c=("en", None), d=("instrumental", None))
    return idx  # e has no language at all


def test_languages_filter_matches_language_or_alt(idx):
    langs_idx(idx)
    q = unit(1, 0)
    assert sorted(paths(rank(idx, q, 9, Filters(languages=["ja"])))) == ["a", "b"]
    assert sorted(paths(rank(idx, q, 9, Filters(languages=["en"])))) == ["b", "c"]
    assert sorted(paths(rank(idx, q, 9, Filters(languages=["en", "ja"])))) == ["a", "b", "c"]


def test_instrumental_filter(idx):
    langs_idx(idx)
    assert paths(rank(idx, unit(1, 0), 9, Filters(languages=["instrumental"]))) == ["d"]


def test_unknown_language_code_is_an_error():
    with pytest.raises(ValueError, match="'xx'"):
        filters_from({"languages": ["ja", "xx"]})
    assert filters_from({"languages": ["JA", "jw"]}).languages == ["ja", "id"]
