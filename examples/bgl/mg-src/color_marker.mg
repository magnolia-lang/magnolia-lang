package examples.bgl.mg-src.color_marker;

concept ColorMarker = {
    type Color;

    function white(): Color;
    function gray(): Color;
    function black(): Color;

    axiom threeDistinctColors() {
        assert white() != gray();
        assert gray() != black();
        assert black() != white();
    }

    axiom exactlyThreeColors(c: Color) {
        assert c == white() || c == gray() || c == black();
    }
}
