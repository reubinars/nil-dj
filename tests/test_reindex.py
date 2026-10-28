import sqlite3

from nil_dj.reindex import log_failures


def test_log_failures_never_raises(capsys):
    class Broken:
        def errors(self):
            raise sqlite3.OperationalError("database is locked")

    log_failures(Broken())
    assert "database is locked" in capsys.readouterr().err
