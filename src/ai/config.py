# Central AI configuration and difficulty presets

from copy import deepcopy

# Baseline defaults
DEFAULT_PARAMS = {
    "minimax_ab": {
        "depth": 4,
        "use_alpha_beta": True,
        "heuristic": "simple",
    },
    "mcts": {
        "simulations": 800,
        "uct_c": 1.4,
        "time_limit": 0.7,
    },
    "q_learning": {
        "enabled": True,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
    },
}

# Difficulty presets adjust all agents coherently
DIFFICULTY_PRESETS = {
    "easy": {
        "minimax_ab": {"depth": 2, "use_alpha_beta": True, "heuristic": "simple"},
        "mcts": {"simulations": 400, "uct_c": 1.6, "time_limit": 0.3},
        "q_learning": {"enabled": False, "alpha": 0.1, "gamma": 0.95, "epsilon": 0.2},
    },
    "medium": {
        "minimax_ab": {"depth": 4, "use_alpha_beta": True, "heuristic": "simple"},
        "mcts": {"simulations": 1500, "uct_c": 1.3, "time_limit": 0.7},
        "q_learning": {"enabled": False, "alpha": 0.1, "gamma": 0.99, "epsilon": 0.1},
    },
    "hard": {
        "minimax_ab": {"depth": 6, "use_alpha_beta": True, "heuristic": "improved"},
        "mcts": {"simulations": 4000, "uct_c": 1.1, "time_limit": 1.5},
        "q_learning": {"enabled": True, "alpha": 0.05, "gamma": 0.995, "epsilon": 0.08},
    },
    "extreme": {
        "minimax_ab": {"depth": 8, "use_alpha_beta": True, "heuristic": "improved"},
        "mcts": {"simulations": 8000, "uct_c": 1.0, "time_limit": 3.0},
        "q_learning": {"enabled": True, "alpha": 0.03, "gamma": 0.997, "epsilon": 0.05},
    },
}


def default_params_copy():
    return {k: deepcopy(v) for k, v in DEFAULT_PARAMS.items()}


def params_for_difficulty(name: str):
    base = DIFFICULTY_PRESETS.get(name, DEFAULT_PARAMS)
    return {k: deepcopy(v) for k, v in base.items()}
