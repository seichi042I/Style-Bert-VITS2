from style_bert_vits2.tts_model import _split_text_to_infer_units


def test_split_preserves_newline_counts():
    assert _split_text_to_infer_units("a\nb", max_unit_chars=100) == [("a", 1), ("b", 0)]
    assert _split_text_to_infer_units("a\n\nb", max_unit_chars=100) == [
        ("a", 2),
        ("b", 0),
    ]


def test_split_long_line_by_fixed_length():
    text = "x" * 25
    units = _split_text_to_infer_units(text, max_unit_chars=10)
    assert [u for u, _ in units] == ["x" * 10, "x" * 10, "x" * 5]
    assert [n for _, n in units] == [0, 0, 0]


def test_split_long_line_with_trailing_newlines():
    text = ("x" * 25) + "\n\n"
    units = _split_text_to_infer_units(text, max_unit_chars=10)
    assert [u for u, _ in units] == ["x" * 10, "x" * 10, "x" * 5]
    assert [n for _, n in units] == [0, 0, 2]


