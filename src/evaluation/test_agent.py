import pytest

from src.environment.main import WebshopAgent


SEARCH_PAGE_ACTIONS = {
    "has_search_bar": True,
    "clickables": ["B001", "next >", "search"],
}


def test_parse_action_extracts_search_query():
    action = WebshopAgent.parse_action(
        "I will search now: search[blue headphones]",
        SEARCH_PAGE_ACTIONS,
    )

    assert action == "search[blue headphones]"


def test_parse_action_normalizes_clickable_to_lowercase():
    action = WebshopAgent.parse_action(
        "click[B001]",
        SEARCH_PAGE_ACTIONS,
    )

    assert action == "click[b001]"


def test_parse_action_supports_search_without_brackets():
    action = WebshopAgent.parse_action(
        "search: red running shoes",
        SEARCH_PAGE_ACTIONS,
    )

    assert action == "search[red running shoes]"


def test_parse_action_prefers_longest_clickable_match():
    actions = {
        "has_search_bar": False,
        "clickables": ["back", "back to search"],
    }

    action = WebshopAgent.parse_action(
        "Please click back to search.",
        actions,
    )

    assert action == "click[back to search]"


def test_parse_action_returns_empty_click_for_unrecognized_reply():
    action = WebshopAgent.parse_action(
        "I do not know what to do.",
        SEARCH_PAGE_ACTIONS,
    )

    assert action == "click[]"


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {"type": "product", "title": "Blue headphones", "price": 19.99},
            "  step 2: click[b001] -> Blue headphones ($19.99)",
        ),
        (
            {"type": "option", "option": "color", "value": "blue"},
            "  step 3: click[blue] -> option color = blue",
        ),
        ({"type": "nav", "target": "next >"}, "  step 4: click[next >]"),
    ],
)
def test_format_step(details, expected):
    assert WebshopAgent.format_step(
        int(expected.split("step ")[1].split(":")[0]),
        expected.split(": ", 1)[1].split(" ->", 1)[0],
        details,
    ) == expected
