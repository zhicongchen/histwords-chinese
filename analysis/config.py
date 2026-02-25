"""Configuration: paths, plot settings, and word lists."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path("Models")
OUTPUTS_DIR = Path("Outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
sns.set(context="notebook", style="whitegrid")

try:
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "SimHei",
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ---------------------------------------------------------------------------
# Word lists — Japan-related terms used in WEFAT analysis
# ---------------------------------------------------------------------------
NATION_STATE_WORDS = [
    "日本", "日本国", "日本政府", "日军", "日本当局", "日方", "日本国家",
    "日本体制", "日方代表", "日政府", "日本官方", "日本军国主义", "日本政权",
]

PEOPLE_WORDS = [
    "日本人", "日本人民", "日侨", "日裔", "日寇", "日籍", "日本公民",
    "倭人", "倭奴", "倭子", "倭寇", "和人", "和族", "小日本", "日人",
    "日本鬼子", "东瀛人", "扶桑人",
]
