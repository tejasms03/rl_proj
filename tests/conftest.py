import pytest


@pytest.fixture(autouse=True)
def _isolated_logdir(tmp_path, monkeypatch):
    # GRF writes dumps/traces relative to the working directory.
    monkeypatch.chdir(tmp_path)
