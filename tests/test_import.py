from ds_prep import hello


def test_ds_prep_import() -> None:
    hello_str = hello()
    assert hello_str == "Hello from ds-prep!"
