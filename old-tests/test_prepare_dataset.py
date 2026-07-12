from src.data.prepare_dataset import clean_text, generate_visitor_rating


def test_clean_text_removes_html_and_normalizes_spaces():
    assert clean_text("Nice<br/>   clean\nplace") == "Nice clean place"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""


def test_generate_visitor_rating_is_bounded_and_decimal():
    rating = generate_visitor_rating(
        "Very clean place near the center, perfect for family.",
        cleanliness=1,
        location=1,
        luxury=0,
        family_friendly=1,
        index=7,
    )

    assert 1.0 <= rating <= 5.0
    assert round(rating, 1) == rating


def test_generate_visitor_rating_is_deterministic_for_same_input():
    first = generate_visitor_rating(
        "Spotless apartment close to museums.",
        cleanliness=1,
        location=1,
        luxury=0,
        family_friendly=0,
        index=11,
    )
    second = generate_visitor_rating(
        "Spotless apartment close to museums.",
        cleanliness=1,
        location=1,
        luxury=0,
        family_friendly=0,
        index=11,
    )

    assert first == second
