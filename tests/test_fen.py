from schaakgui.fen import (GameInfo, check_position, full_fen, placement_fen, squares_from_fen,
                           suggest_castling)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_round_trip_start_position():
    squares = squares_from_fen(START + " w KQkq - 0 1")
    assert len(squares) == 32 and squares["e1"] == "K" and squares["d8"] == "q"
    assert placement_fen(squares) == START
    assert full_fen(squares, GameInfo(castling="KQkq")) == START + " w KQkq - 0 1"
    assert placement_fen({}) == "8/8/8/8/8/8/8/8"


def test_suggest_castling():
    assert suggest_castling(squares_from_fen(START)) == "KQkq"
    squares = squares_from_fen(START)
    del squares["h1"], squares["a8"]
    assert suggest_castling(squares) == "Qk"
    squares.pop("e8")
    assert suggest_castling(squares) == "Q"
    assert suggest_castling({}) == "-"


def test_check_position_is_quiet_for_legal_position():
    assert check_position(squares_from_fen(START), GameInfo(castling="KQkq")) == []


def test_check_position_warnings():
    missing = squares_from_fen("8/8/8/8/8/8/8/4K3")
    assert any("black has no king" in w.lower() for w in check_position(missing))

    pawn = squares_from_fen("4k3/8/8/8/8/8/8/P3K3")
    assert any("pawn" in w.lower() for w in check_position(pawn))

    # Black is in check while it is White's move: the side not to move is in check.
    check = squares_from_fen("4k3/8/8/8/8/8/8/4KR2")
    check["f8"] = "R"
    del check["f1"]
    warnings = check_position(check, GameInfo(side="w"))
    assert any("check" in w.lower() for w in warnings)
    # the input is never changed
    assert check["f8"] == "R"
