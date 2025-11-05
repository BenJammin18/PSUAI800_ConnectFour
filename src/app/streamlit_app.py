import sys
from pathlib import Path as _Path
# Ensure Streamlit and utilities are imported
import streamlit as st
import random
from pathlib import Path
import importlib

ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.config import default_params_copy, params_for_difficulty

# Optional AI modules
try:
    from src.ai import mcts as ai_mcts
except Exception:
    ai_mcts = None
try:
    from src.ai import minmax as ai_minmax
except Exception:
    ai_minmax = None
try:
    from src.ai import rl_agent as ai_qlearning
except Exception:
    ai_qlearning = None


def reload_ai_modules():
    # Import once; avoid reload loops that can destabilize Streamlit runtime
    global ai_mcts, ai_minmax, ai_qlearning
    try:
        if ai_mcts is None:
            ai_mcts = importlib.import_module("src.ai.mcts")
    except Exception as e:
        ai_mcts = None
        st.session_state["last_ai_error"] = f"Failed to load MCTS: {e}"
    try:
        if ai_minmax is None:
            ai_minmax = importlib.import_module("src.ai.minmax")
    except Exception as e:
        ai_minmax = None
        st.session_state["last_ai_error"] = f"Failed to load Minimax: {e}"
    try:
        if ai_qlearning is None:
            ai_qlearning = importlib.import_module("src.ai.rl_agent")
    except Exception:
        ai_qlearning = None


## Initial State and utilities


def ensure_state():
    s = st.session_state
    s.setdefault("rows", 6)
    s.setdefault("cols", 7)
    s.setdefault("win_len", 4)
    s.setdefault("gravity", True)
    # Ensure board exists with correct shape
    if (
        "board" not in s
        or not isinstance(s.board, list)
        or len(s.board) != s.rows
        or (s.rows > 0 and (not isinstance(s.board[0], list) or len(s.board[0]) != s.cols))
    ):
        s.board = new_board(s.rows, s.cols)
    s.setdefault("player", 1)
    s.setdefault("game_over", False)
    s.setdefault("winner", 0)
    s.setdefault("winning_cells", [])
    s.setdefault("move_id", 0)
    s.setdefault("last_move_cell", None)
    s.setdefault("player_types", {1: "human", 2: "program"})
    s.setdefault("ai_strategies", ["minimax_ab", "mcts", "q_learning"])  # order = preference
    if "ai_params" not in s:
        s.ai_params = default_params_copy()
    s.setdefault("difficulty", "medium")
    s.setdefault("last_applied_difficulty", s["difficulty"])  # track last applied preset
    # Initialize AI status keys
    s.setdefault("ai_thinking", False)
    s.setdefault("ai_policy_choice", None)
    s.setdefault("last_ai_error", None)


def new_board(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def reset_game(rows, cols, win_len, gravity, start_player=1, p1_type=None, p2_type=None):
    s = st.session_state
    s.rows, s.cols, s.win_len, s.gravity = int(rows), int(cols), int(win_len), bool(gravity)
    s.board = new_board(s.rows, s.cols)
    s.player = int(start_player)
    s.game_over = False
    s.winner = 0
    s.winning_cells = []
    s.move_id = 0
    s.last_move_cell = None
    s.ai_thinking = False
    s.ai_policy_choice = None
    if p1_type is not None:
        s.player_types[1] = p1_type
    if p2_type is not None:
        s.player_types[2] = p2_type
    apply_board_css()


def legal_moves(board, gravity):
    rows = len(board)
    cols = len(board[0]) if rows else 0
    if gravity:
        out = []
        for c in range(cols):
            for r in range(rows - 1, -1, -1):
                if board[r][c] == 0:
                    out.append(c)
                    break
        return out
    else:
        return [(c, r) for r in range(rows) for c in range(cols) if board[r][c] == 0]


def apply_move(board, player, move, gravity):
    # Accept both (col,row) and column-int moves. If a column int is provided while gravity is False,
    # fall back to dropping into the lowest available row in that column to avoid runtime errors.
    if gravity or isinstance(move, int):
        col = int(move) if not isinstance(move, tuple) else move  # ensure int column
        if isinstance(col, tuple):
            # Defensive: if a tuple slipped through, unpack normally
            col, row = col
            if board[row][col] == 0:
                board[row][col] = player
                return row, col
            return None
        for r in range(len(board) - 1, -1, -1):
            if board[r][col] == 0:
                board[r][col] = player
                return r, col
        return None
    else:
        col, row = move
        if board[row][col] == 0:
            board[row][col] = player
            return row, col
        return None


def check_win(board, win_len):
    rows = len(board)
    cols = len(board[0]) if rows else 0
    dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(rows):
        for c in range(cols):
            p = board[r][c]
            if p == 0:
                continue
            for dr, dc in dirs:
                cells = [(r, c)]
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == p:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == p:
                    cells.insert(0, (rr, cc))
                    rr -= dr
                    cc -= dc
                if len(cells) >= win_len:
                    return p, cells[:win_len] if len(cells) > win_len else cells
    return 0, []


def cell_display_val(v, is_winning, is_last):
    if v == 0:
        return ""
    if is_winning:
        return "🟩"
    return "🔴" if v == 1 else "🟡"


## AI Player helpers

def _try_call(module, names, *args, **kwargs):
    if module is None:
        return None
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                st.session_state["last_ai_error"] = f"{module.__name__}.{n} failed: {e}"
                return None
    return None


def _has_api(module, names):
    if module is None:
        return False
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn):
            return True
    return False


def _strategy_available(strat):
    s = st.session_state
    if strat in ("mcts", "mcts_uct"):
        return ai_mcts is not None and any(callable(getattr(ai_mcts, n, None)) for n in ["get_move", "uct_search", "search", "MCTS"])
    if strat == "minimax_ab":
        if ai_minmax is None:
            return False
        if any(callable(getattr(ai_minmax, n, None)) for n in ["get_move", "best_move"]):
            return True
        # Class API
        cls = getattr(ai_minmax, "Minimax", None)
        return cls is not None
    if strat == "q_learning":
        return ai_qlearning is not None and any(callable(getattr(ai_qlearning, n, None)) for n in ["get_move", "policy_move", "act"]) and s.ai_params.get("q_learning", {}).get("enabled")
    return False


def choose_ai_strategy():
    s = st.session_state
    # Normalize selected to a supported module
    selected = s.ai_strategies[0] if s.ai_strategies else "mcts"
    if selected in ("mcts", "mcts_uct") and _strategy_available("mcts"):
        return "mcts"
    if selected == "minimax_ab" and _strategy_available("minimax_ab"):
        return "minimax_ab"
    if selected == "q_learning" and _strategy_available("q_learning"):
        return "q_learning"
    # Fallback order
    for cand in ("mcts", "minimax_ab", "q_learning"):
        if _strategy_available(cand):
            return cand
    return None


def get_ai_move(board, player, gravity):
    s = st.session_state
    strat = choose_ai_strategy()
    if strat is None:
        moves = legal_moves(board, gravity)
        return random.choice(moves) if moves else None
    s.ai_policy_choice = strat
    params = s.ai_params.get(strat, {})
    if strat == "minimax_ab" and ai_minmax is not None:
        # Prefer function; fallback to class
        mv = _try_call(ai_minmax, ["get_move", "best_move"], board, player, params)
        if mv is None:
            MM = getattr(ai_minmax, "Minimax", None)
            if MM is not None:
                try:
                    mv = MM(board, player, params).search()
                except Exception:
                    mv = None
        if mv is not None:
            return mv
    if strat in ("mcts", "mcts_uct") and ai_mcts is not None:
        if ai_mcts is None:
            s.last_ai_error = "MCTS module not available"
            moves = legal_moves(board, gravity)
            return random.choice(moves) if moves else None
        base = s.ai_params.get("mcts", {})
        sims = int(base.get("simulations", 400))
        sims = max(200, min(sims, 5000))  # defensive cap
        cval = float(base.get("uct_c", 1.4))
        tlimit = float(base.get("time_limit", 0.7) or 0.7)
        mcts_params = {
            "simulations": sims,
            "uct_c": cval,
            "win_len": s.win_len,
            "gravity": s.gravity,
            "time_limit": tlimit,
        }
        mv = _try_call(ai_mcts, ["get_move", "uct_search", "search"], board, player, mcts_params)
        if mv is not None:
            return mv
    if strat == "q_learning" and ai_qlearning is not None:
        mv = _try_call(ai_qlearning, ["get_move", "policy_move", "act"], board, player, params)
        if mv is not None:
            return mv
    moves = legal_moves(board, gravity)
    return random.choice(moves) if moves else None


def make_ai_move():
    ensure_state()
    s = st.session_state
    if s.game_over:
        return
    s.ai_thinking = True
    mv = get_ai_move(s.board, s.player, s.gravity)
    s.ai_thinking = False
    if mv is None:
        return
    res = apply_move(s.board, s.player, mv, s.gravity)
    if res is None:
        return
    r, c = res
    s.last_move_cell = (r, c)
    s.move_id += 1
    winner, cells = check_win(s.board, s.win_len)
    if winner:
        s.game_over = True
        s.winner = winner
        s.winning_cells = cells
    else:
        full = all(s.board[rr][cc] != 0 for rr in range(s.rows) for cc in range(s.cols))
        if full:
            s.game_over = True
            s.winner = 0
        else:
            s.player = 2 if s.player == 1 else 1


## Interaction for Game play

def place_token(col, row=None):
    ensure_state()
    s = st.session_state
    if s.game_over:
        return
    board = s.board
    player = s.player
    mv = col if s.gravity else (col, row)
    res = apply_move(board, player, mv, s.gravity)
    if res is None:
        return
    r, c = res
    s.last_move_cell = (r, c)
    s.move_id += 1
    winner, cells = check_win(board, s.win_len)
    if winner:
        s.game_over = True
        s.winner = winner
        s.winning_cells = cells
        return
    full = all(board[rr][cc] != 0 for rr in range(s.rows) for cc in range(s.cols))
    if full:
        s.game_over = True
        s.winner = 0
        return
    s.player = 2 if player == 1 else 1
    if s.autoplay_ai and s.player_types.get(s.player) == "program":
        make_ai_move()



## Styling

def apply_board_css():
    s = st.session_state
    cols, rows = int(s.cols), int(s.rows)
    max_total = 560
    min_cell = 36
    preferred = max(min_cell, min(64, int(max_total / max(1, cols))))
    cells_gap = 10 if cols <= 10 else 6
    css_vars = f":root{{--board-columns:{cols};--board-rows:{rows};--cell-size:{preferred}px;--cells-gap:{cells_gap}px;}}"
    st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)



## UI

st.set_page_config(page_title="Connect Game", layout="wide")
st.title("Connect Game — Classic Style")

## Inject CSS from file right after title
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    try:
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        pass

ensure_state()
reload_ai_modules()
apply_board_css()

## Configurations
with st.expander("Configurations", expanded=True):
    left, right = st.columns(2)
    with left:
        rows = st.number_input("Rows", 4, 12, st.session_state.rows, 1)
        cols = st.number_input("Columns", 4, 12, st.session_state.cols, 1)
        win_len = st.selectbox("Win condition (connect)", [4, 5], index=0 if st.session_state.win_len == 4 else 1)
        gravity = st.checkbox("Gravity (classic drop)", value=st.session_state.gravity)
        first_player = st.selectbox("First player", ["Red (1)", "Yellow (2)"], index=0 if st.session_state.player == 1 else 1)
        start_player = 1 if first_player.startswith("Red") else 2
        # Difficulty
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard", "extreme"], index=["easy","medium","hard","extreme"].index(st.session_state.get("difficulty","medium")))
        st.session_state.difficulty = difficulty
        # Apply preset immediately if difficulty changed
        if st.session_state.difficulty != st.session_state.last_applied_difficulty:
            st.session_state.ai_params = params_for_difficulty(st.session_state.difficulty)
            st.session_state.last_applied_difficulty = st.session_state.difficulty
    with right:
        p1_sel = st.selectbox("Player 1 (Red)", ["Human", "AI"], index=0 if st.session_state.player_types.get(1)=="human" else 1)
        p2_sel = st.selectbox("Player 2 (Yellow)", ["Human", "AI"], index=0 if st.session_state.player_types.get(2)=="human" else 1)
        # Apply player type selections immediately
        st.session_state.player_types[1] = "human" if p1_sel == "Human" else "program"
        st.session_state.player_types[2] = "human" if p2_sel == "Human" else "program"
        # AI strategy preference — show only available ones
        options = []
        if _strategy_available("mcts"):
            options.append("mcts")
        if _strategy_available("minimax_ab"):
            options.append("minimax_ab")
        if _strategy_available("q_learning"):
            options.append("q_learning")
        if not options:
            options = ["mcts"]
        cur_pref = st.session_state.ai_strategies[0] if st.session_state.ai_strategies else options[0]
        if cur_pref == "mcts_uct":
            cur_pref = "mcts"
        pref = st.selectbox("AI strategy", options, index=options.index(cur_pref) if cur_pref in options else 0)
        st.session_state.ai_strategies = [pref] + [x for x in options if x != pref]
        autoplay = st.checkbox("Autoplay AI moves", value=st.session_state.get("autoplay_ai", True), key="autoplay_ai")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        start = st.button("Start", key="start_btn")
        new = st.button("New Game", key="new_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        if start or new:
            st.session_state.ai_params = params_for_difficulty(st.session_state.difficulty)
            reset_game(rows, cols, win_len, gravity, start_player,
                       p1_type=("human" if p1_sel == "Human" else "program"),
                       p2_type=("human" if p2_sel == "Human" else "program"))

## Early AI first move if autoplay and program starts
if (
    st.session_state.autoplay_ai and
    st.session_state.player_types.get(st.session_state.player) == "program" and
    st.session_state.move_id == 0 and not st.session_state.game_over
):
    make_ai_move()

# Begin board scope wrapper for styling
st.markdown('<div class="board-scope">', unsafe_allow_html=True)

## Drop buttons
st.markdown('<div class="drop-row">', unsafe_allow_html=True)
if st.session_state.gravity:
    drop_cols = st.columns(st.session_state.cols)
    for ci in range(st.session_state.cols):
        with drop_cols[ci]:
            if st.button("⬇", key=f"drop_{ci}_{st.session_state.move_id}"):
                place_token(ci)
st.markdown('</div>', unsafe_allow_html=True)

## Grid
st.markdown('<div class="grid">', unsafe_allow_html=True)
cols_n = st.session_state.cols
rows_n = st.session_state.rows
board = st.session_state.board
win_set = set(st.session_state.winning_cells)
last = st.session_state.get("last_move_cell")

for r in range(rows_n):
    row_cols = st.columns(cols_n)
    for c in range(cols_n):
        with row_cols[c]:
            is_win = (r, c) in win_set
            is_last = (r, c) == tuple(last) if last else False
            disp = cell_display_val(board[r][c], is_win, is_last)
            key = f"cell_{r}_{c}_{st.session_state.move_id}"
            base_tag = "cell-empty" if board[r][c] == 0 else ("cell-p1" if board[r][c] == 1 else "cell-p2")
            if is_win:
                base_tag += " win"
            if is_last:
                base_tag += " last"
            if st.session_state.gravity:
                # Render as disabled circular token (non-clickable)
                st.button(disp, key=key, help=base_tag, disabled=True)
            else:
                if board[r][c] == 0:
                    if st.button(disp, key=key, help=base_tag):
                        place_token(c, r)
                else:
                    # Occupied cell shown as disabled token
                    st.button(disp, key=key, help=base_tag, disabled=True)

st.markdown('</div>', unsafe_allow_html=True)

# End board scope wrapper
st.markdown('</div>', unsafe_allow_html=True)

## Status panels
left_p, right_p = st.columns(2)
with left_p:
    st.markdown('<div class="panel"><div class="panel-title">Game Status</div>', unsafe_allow_html=True)
    if st.session_state.game_over:
        if st.session_state.winner == 0:
            st.success("Game over: Tie!")
        else:
            st.success("Game over: Red (1) wins!" if st.session_state.winner == 1 else "Game over: Yellow (2) wins!")
    else:
        badge = 'Red (1)' if st.session_state.player == 1 else 'Yellow (2)'
        st.info(f"Current turn: {badge}")
    if st.session_state.last_move_cell:
        rr, cc = st.session_state.last_move_cell
        st.caption(f"Last move: row {rr+1}, col {cc+1}")
    st.caption(f"Moves made: {st.session_state.move_id}")
    st.markdown('</div>', unsafe_allow_html=True)
with right_p:
    st.markdown('<div class="panel"><div class="panel-title">AI Config</div>', unsafe_allow_html=True)
    strat = st.session_state.ai_policy_choice
    st.caption(f"Strategy selected: {strat}")
    if st.session_state.get("last_ai_error"):
        st.error(f"AI error: {st.session_state.last_ai_error}")
    if strat:
        if strat in ("mcts", "mcts_uct"):
            st.info("Using AI: Monte Carlo Tree Search (MCTS)")
        elif strat == "minimax_ab":
            st.info("Using AI: Minimax with Alpha-Beta Pruning")
        elif strat == "q_learning":
            st.info("Using AI: Q-Learning")
        else:
            st.info("AI strategy not recognized")
    else:
        st.info("No AI strategy selected")
    st.markdown('</div>', unsafe_allow_html=True)