from src import crash_logger


def test_install_idempotent():
    p1 = crash_logger.install()
    p2 = crash_logger.install()
    # Either both succeed and equal, or both None on a system where logging
    # can't write (very rare in CI).
    assert p1 == p2


def test_log_dir_exists():
    p = crash_logger.crash_log_path()
    if p is not None:
        assert p.parent.exists()
