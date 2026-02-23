import streamlit as st
import chess
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ----------------------------
# Capture Chess Variant
# ----------------------------
# Convert Capture:
# If a normal capture move is legal (per python-chess),
# player may instead "convert capture":
#   - The moving piece DOES NOT move
#   - The captured enemy piece at target square is replaced by a friendly piece of the SAME TYPE
#   - Turn switches
#
# We restrict convert-capture to only those captures that are legal as normal captures
# so we keep king safety/pins/check rules consistent and avoid ambiguous edge cases.
# ----------------------------

PIECE_TO_UNICODE = {
    chess.PAWN:   ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK:   ("♖", "♜"),
    chess.QUEEN:  ("♕", "♛"),
    chess.KING:   ("♔", "♚"),
}

FILES = "abcdefgh"
RANKS = "12345678"

LIGHT = "#F0D9B5"
DARK  = "#B58863"
SEL   = "#FFD166"  # selected square highlight
MOVEH = "#8EECF5"  # legal move highlight
CAPTH = "#FF5C8A"  # capture target highlight


@dataclass
class ActionRecord:
    fen_before: str
    fen_after: str
    description: str


def square_name(sq: int) -> str:
    return chess.square_name(sq)


def piece_unicode(piece: Optional[chess.Piece]) -> str:
    if piece is None:
        return " "
    white_sym, black_sym = PIECE_TO_UNICODE[piece.piece_type]
    return white_sym if piece.color == chess.WHITE else black_sym


def init_state():
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "selected" not in st.session_state:
        st.session_state.selected = None  # selected square int
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []  # List[ActionRecord]
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []  # List[ActionRecord]
    if "status" not in st.session_state:
        st.session_state.status = ""
    if "capture_mode" not in st.session_state:
        st.session_state.capture_mode = "Convert (don’t move)"
    if "vs_ai" not in st.session_state:
        st.session_state.vs_ai = False
    if "ai_color" not in st.session_state:
        st.session_state.ai_color = chess.BLACK
    if "ai_style" not in st.session_state:
        st.session_state.ai_style = "Balanced"
    if "last_move_squares" not in st.session_state:
        st.session_state.last_move_squares = None  # (from,to) squares for highlighting


def reset_game():
    st.session_state.board = chess.Board()
    st.session_state.selected = None
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []
    st.session_state.status = ""
    st.session_state.last_move_squares = None


def legal_moves_from(board: chess.Board, from_sq: int) -> List[chess.Move]:
    return [m for m in board.legal_moves if m.from_square == from_sq]


def is_capture(board: chess.Board, move: chess.Move) -> bool:
    return board.is_capture(move)


def find_legal_move(board: chess.Board, from_sq: int, to_sq: int, promotion_piece: Optional[int] = None) -> Optional[chess.Move]:
    for m in board.legal_moves:
        if m.from_square == from_sq and m.to_square == to_sq:
            if promotion_piece is None:
                # if move is a promotion, python-chess encodes m.promotion
                if m.promotion is None:
                    return m
            else:
                if m.promotion == promotion_piece:
                    return m
    return None


def apply_normal_move(board: chess.Board, move: chess.Move) -> str:
    # Returns description
    san = board.san(move)
    board.push(move)
    return f"Normal: {san}"


def apply_convert_capture(board: chess.Board, move: chess.Move) -> str:
    """
    move must be a legal CAPTURE move in standard chess.
    Convert capture:
      - mover stays put (does not move)
      - captured piece on to_square becomes friendly piece of same type
      - turn toggles, clocks updated
      - en passant target cleared
      - if captured piece is rook etc, we keep castling rights unchanged for simplicity
        (you can optionally tighten this, but it’s coherent as a variant rule)
    """
    assert board.is_legal(move), "Convert capture requires a legal capture move."
    assert board.is_capture(move), "Convert capture requires capture."

    mover = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)

    # En passant capture: captured piece is not on to_square in standard chess.
    # python-chess represents EP as capture; we must handle it.
    ep_capture = board.is_en_passant(move)

    if mover is None:
        raise ValueError("No mover piece on from_square.")
    if captured is None and not ep_capture:
        raise ValueError("No captured piece on to_square (and not en passant).")
    if mover.color != board.turn:
        raise ValueError("Not your turn for that piece.")

    # Determine actual captured piece and square
    if ep_capture:
        # Captured pawn is behind the to_square
        direction = -1 if board.turn == chess.WHITE else 1
        cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.to_square) + direction)
        captured_piece = board.piece_at(cap_sq)
        if captured_piece is None or captured_piece.piece_type != chess.PAWN:
            raise ValueError("Invalid en passant state.")
        captured_type = captured_piece.piece_type
        # Remove captured pawn
        board.remove_piece_at(cap_sq)
        # Place converted pawn on the to_square (same type = pawn) as current player
        board.set_piece_at(move.to_square, chess.Piece(captured_type, board.turn))
    else:
        captured_type = captured.piece_type
        # Replace captured piece with your piece of same type
        board.set_piece_at(move.to_square, chess.Piece(captured_type, board.turn))

    # Mover stays where it is (do nothing to from_square)

    # Clear en passant
    board.ep_square = None

    # Update move counters
    # In standard rules, halfmove clock resets on capture or pawn move.
    board.halfmove_clock = 0
    if board.turn == chess.BLACK:
        board.fullmove_number += 1

    # Toggle turn
    board.turn = not board.turn

    # Description (custom)
    return f"Convert: {piece_unicode(mover)} converts {square_name(move.to_square)} into your {chess.piece_name(captured_type)}"


def record_action(fen_before: str, fen_after: str, desc: str):
    st.session_state.undo_stack.append(ActionRecord(fen_before, fen_after, desc))
    st.session_state.redo_stack = []  # clear redo on new action


def undo():
    if not st.session_state.undo_stack:
        return
    rec = st.session_state.undo_stack.pop()
    st.session_state.redo_stack.append(rec)
    st.session_state.board.set_fen(rec.fen_before)
    st.session_state.selected = None
    st.session_state.status = f"Undid: {rec.description}"
    st.session_state.last_move_squares = None


def redo():
    if not st.session_state.redo_stack:
        return
    rec = st.session_state.redo_stack.pop()
    st.session_state.undo_stack.append(rec)
    st.session_state.board.set_fen(rec.fen_after)
    st.session_state.selected = None
    st.session_state.status = f"Redid: {rec.description}"
    st.session_state.last_move_squares = None


def game_status(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate — {winner} wins."
    if board.is_stalemate():
        return "Stalemate — draw."
    if board.is_insufficient_material():
        return "Draw — insufficient material."
    if board.can_claim_threefold_repetition():
        return "Threefold repetition can be claimed."
    if board.can_claim_fifty_moves():
        return "50-move rule can be claimed."
    if board.is_check():
        return "Check!"
    return "In progress."


def choose_ai_action(board: chess.Board, style: str) -> Tuple[str, Optional[chess.Move]]:
    """
    AI chooses either:
      - normal move
      - convert capture (if capture exists and style prefers it)
    """
    legal = list(board.legal_moves)
    if not legal:
        return ("AI has no legal moves.", None)

    captures = [m for m in legal if board.is_capture(m)]

    # Simple heuristics:
    # - "Convert-heavy": prefers convert-captures whenever possible
    # - "Aggressive": prefers normal captures (gaining space)
    # - "Balanced": picks between normal and convert based on piece value and safety-ish
    piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

    if style == "Convert-heavy" and captures:
        m = random.choice(captures)
        return ("convert", m)

    if style == "Aggressive" and captures:
        # prefer capturing higher value targets normally
        captures_sorted = sorted(
            captures,
            key=lambda m: piece_value.get((board.piece_at(m.to_square).piece_type if not board.is_en_passant(m) else chess.PAWN), 0),
            reverse=True
        )
        return ("normal", captures_sorted[0])

    if style == "Balanced":
        if captures:
            # choose between normal vs convert by comparing "gain":
            # normal: you move and remove enemy piece (material trade = +value captured)
            # convert: you keep your piece AND you gain a new piece (roughly +value captured, but you don't advance)
            # We'll bias convert for big pieces, normal for pawns/minors to keep initiative.
            m = max(
                captures,
                key=lambda mv: piece_value.get((board.piece_at(mv.to_square).piece_type if not board.is_en_passant(mv) else chess.PAWN), 0)
            )
            captured_type = (board.piece_at(m.to_square).piece_type if not board.is_en_passant(m) else chess.PAWN)
            if piece_value.get(captured_type, 0) >= 5:
                return ("convert", m)
            else:
                # 50/50 for small captures
                return (random.choice(["normal", "convert"]), m)

    # Otherwise random legal move
    return ("normal", random.choice(legal))


def maybe_ai_move():
    board = st.session_state.board
    if not st.session_state.vs_ai:
        return
    if board.is_game_over():
        return
    if board.turn != st.session_state.ai_color:
        return

    fen_before = board.fen()
    kind, move = choose_ai_action(board, st.session_state.ai_style)
    if move is None:
        return

    try:
        if kind == "convert" and board.is_capture(move):
            desc = apply_convert_capture(board, move)
            st.session_state.last_move_squares = (move.from_square, move.to_square)
        else:
            desc = apply_normal_move(board, move)
            st.session_state.last_move_squares = (move.from_square, move.to_square)
    except Exception as e:
        # fallback to normal move
        board.set_fen(fen_before)
        desc = apply_normal_move(board, move)
        st.session_state.last_move_squares = (move.from_square, move.to_square)

    fen_after = board.fen()
    record_action(fen_before, fen_after, f"AI: {desc}")
    st.session_state.status = f"AI played — {desc}"


def render_board(board: chess.Board, selected: Optional[int], legal_to: List[int], capture_to: List[int], last_move: Optional[Tuple[int,int]]):
    # Draw as an 8x8 grid of buttons (simple + reliable on Streamlit Cloud)
    # Rank 8 at top, file a at left.
    for rank in range(7, -1, -1):
        cols = st.columns(8, gap="small")
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            label = piece_unicode(piece)

            base = LIGHT if (file + rank) % 2 == 0 else DARK
            bg = base

            if last_move and (sq == last_move[0] or sq == last_move[1]):
                bg = "#C7F9CC"  # last-move highlight

            if selected == sq:
                bg = SEL
            elif sq in capture_to:
                bg = CAPTH
            elif sq in legal_to:
                bg = MOVEH

            # make buttons with CSS background (Streamlit needs a bit of trickery)
            # We'll render a small HTML-ish style with markdown is unsafe for buttons, so use st.button + CSS classes is hard.
            # Instead: show color via emoji square in label prefix for clarity.
            prefix = "🟥 " if bg == CAPTH else ("🟦 " if bg == MOVEH else ("🟨 " if bg == SEL else ("🟩 " if bg == "#C7F9CC" else "")))
            btn_label = f"{prefix}{label}"

            if cols[file].button(btn_label, key=f"sq_{sq}", use_container_width=True):
                on_square_click(sq)


def on_square_click(sq: int):
    board = st.session_state.board
    selected = st.session_state.selected

    if board.is_game_over():
        st.session_state.status = "Game is over. Reset to play again."
        return

    # If nothing selected: select your own piece
    if selected is None:
        p = board.piece_at(sq)
        if p is None:
            st.session_state.status = "Select one of your pieces."
            return
        if p.color != board.turn:
            st.session_state.status = "It’s not that side’s turn."
            return
        st.session_state.selected = sq
        st.session_state.status = f"Selected {square_name(sq)}"
        return

    # If clicking same square: deselect
    if selected == sq:
        st.session_state.selected = None
        st.session_state.status = "Selection cleared."
        return

    # Attempt move from selected to sq
    from_sq = selected
    to_sq = sq

    # Handle pawn promotion choice if needed (normal move)
    mover = board.piece_at(from_sq)
    if mover is None:
        st.session_state.selected = None
        return

    promotion_piece = None
    needs_promo = (
        mover.piece_type == chess.PAWN and
        (chess.square_rank(to_sq) == 7 if board.turn == chess.WHITE else chess.square_rank(to_sq) == 0)
    )

    # If promotion possible, we’ll prefer Queen by default and allow override in UI via sidebar
    if needs_promo:
        promo_map = {"Queen": chess.QUEEN, "Rook": chess.ROOK, "Bishop": chess.BISHOP, "Knight": chess.KNIGHT}
        promotion_piece = promo_map.get(st.session_state.get("promo_choice", "Queen"), chess.QUEEN)

    move = find_legal_move(board, from_sq, to_sq, promotion_piece=promotion_piece)
    if move is None:
        # if promotion not specified, try any promotion move and prompt user
        if needs_promo:
            any_promo = None
            for m in board.legal_moves:
                if m.from_square == from_sq and m.to_square == to_sq and m.promotion is not None:
                    any_promo = m
                    break
            if any_promo:
                st.session_state.status = "Promotion move available — pick promotion piece in sidebar, then click target again."
                return

        st.session_state.status = "Illegal move."
        return

    fen_before = board.fen()

    # Decide capture mode
    if board.is_capture(move):
        mode = st.session_state.capture_mode
        try:
            if mode.startswith("Convert"):
                desc = apply_convert_capture(board, move)
            else:
                desc = apply_normal_move(board, move)
        except Exception as e:
            board.set_fen(fen_before)
            st.session_state.status = f"Move failed: {e}"
            st.session_state.selected = None
            return
    else:
        try:
            desc = apply_normal_move(board, move)
        except Exception as e:
            board.set_fen(fen_before)
            st.session_state.status = f"Move failed: {e}"
            st.session_state.selected = None
            return

    fen_after = board.fen()
    record_action(fen_before, fen_after, desc)
    st.session_state.last_move_squares = (from_sq, to_sq)
    st.session_state.selected = None
    st.session_state.status = desc


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Capture Chess", page_icon="♟️", layout="wide")
init_state()

st.title("♟️ Capture Chess")
st.caption("Variant rule: On a legal capture, you may either capture normally OR convert the captured piece into your own piece of the same type without moving.")

# Sidebar controls
with st.sidebar:
    st.header("Game Controls")

    if st.button("Reset Game", use_container_width=True):
        reset_game()

    c1, c2 = st.columns(2)
    if c1.button("Undo", use_container_width=True):
        undo()
    if c2.button("Redo", use_container_width=True):
        redo()

    st.divider()
    st.header("Capture Mode")
    st.session_state.capture_mode = st.radio(
        "When you capture…",
        options=["Convert (don’t move)", "Normal (move onto square)"],
        index=0 if st.session_state.capture_mode.startswith("Convert") else 1
    )

    st.divider()
    st.header("Promotion")
    st.session_state.promo_choice = st.selectbox("Promote pawns to", ["Queen", "Rook", "Bishop", "Knight"], index=0)

    st.divider()
    st.header("Play vs AI")
    st.session_state.vs_ai = st.toggle("Enable AI opponent", value=st.session_state.vs_ai)
    st.session_state.ai_color = chess.BLACK if st.selectbox("AI plays as", ["Black", "White"], index=0) == "Black" else chess.WHITE
    st.session_state.ai_style = st.selectbox("AI style", ["Balanced", "Convert-heavy", "Aggressive"], index=0)

# Main view
board = st.session_state.board
status = game_status(board)

top_left, top_right = st.columns([2, 1], gap="large")

with top_left:
    st.subheader("Board")

    selected = st.session_state.selected
    legal_to = []
    capture_to = []

    if selected is not None:
        for m in legal_moves_from(board, selected):
            legal_to.append(m.to_square)
            if board.is_capture(m):
                capture_to.append(m.to_square)

    render_board(board, selected, legal_to, capture_to, st.session_state.last_move_squares)

with top_right:
    st.subheader("Status")
    turn = "White" if board.turn == chess.WHITE else "Black"
    st.write(f"**Turn:** {turn}")
    st.write(f"**Game:** {status}")
    if st.session_state.status:
        st.info(st.session_state.status)

    st.divider()
    st.subheader("Move Log (this session)")
    if st.session_state.undo_stack:
        for i, rec in enumerate(reversed(st.session_state.undo_stack[-12:]), start=1):
            st.write(f"- {rec.description}")
    else:
        st.write("No moves yet.")

    st.divider()
    st.subheader("Current Position (FEN)")
    st.code(board.fen(), language="text")

# After user move, maybe AI responds
maybe_ai_move()

# If AI moved, refresh status area
st.caption("Tips: Select a piece, then click a target square. Captures are highlighted. Use Undo/Redo to explore tactics.")
