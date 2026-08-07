from sight.tokens import build_design_tokens

DUMP = {
    "colors": [
        {"hex": "#FDFDFD", "ratio": 0.59},
        {"hex": "#262525", "ratio": 0.07},
        {"hex": "#29519E", "ratio": 0.05},
        {"hex": "#989492", "ratio": 0.19},
    ],
    "elements": [
        {"kind": "text", "font": {"fontSize": 33}},
        {"kind": "text", "font": {"fontSize": 15}},
        {"kind": "text", "font": {"fontSize": 33}},
        {"kind": "text", "font": {"fontSize": 71}},
    ],
    "controls": [{"cornerRadius": 26}, {"cornerRadius": 0}, {"cornerRadius": 26}],
    "gaps": [{"px": 16}, {"px": 16}, {"px": 24}],
    "shadows": [],
}


def test_color_roles() -> None:
    tokens = build_design_tokens(DUMP)
    assert tokens["color"]["background"]["$value"] == "#FDFDFD"
    assert tokens["color"]["ink"]["$value"] == "#262525"
    assert tokens["color"]["accent"]["$value"] == "#29519E"  # самый насыщенный


def test_typography_scale_sorted() -> None:
    tokens = build_design_tokens(DUMP)
    assert tokens["typography"]["scale"]["$value"] == [15, 33, 71]


def test_spacing_and_radius() -> None:
    tokens = build_design_tokens(DUMP)
    assert tokens["spacing"]["base"]["$value"] == "16px"
    assert tokens["borderRadius"]["pill"]["$value"] == "26px"
