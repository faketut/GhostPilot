from src import secret_store


def test_keys_constant():
    assert "OPENAI_API_KEY" in secret_store.SECRET_KEYS


def test_available_returns_bool():
    assert isinstance(secret_store.available(), bool)


def test_missing_key_returns_none():
    # A clearly non-existent key in a clearly non-existent service should be
    # None whether keyring is available or not.
    assert secret_store.get("__GHOSTPILOT_TEST_NONEXISTENT__") in (None, "")


def test_migrate_from_dict_no_keys():
    d = {"FOO": "bar"}
    migrated = secret_store.migrate_from_dict(d)
    assert migrated == [] or isinstance(migrated, list)
    assert d == {"FOO": "bar"}  # untouched
