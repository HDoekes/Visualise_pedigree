import streamlit as st
import pandas as pd
import graphviz
import io

st.set_page_config(page_title="Pedigree Viewer", layout="wide")

st.title("🐄 Pedigree Visualiser")
st.caption(
    "Upload a pedigree file (comma- or space-delimited, 3 columns: animal, sire, dam), "
    "pick an animal, and explore its ancestry for N generations. "
    "Ancestors sharing the focal animal's ID are highlighted in red — a sign of a cyclic pedigree."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MISSING_TOKENS = {"", "0", "na", "n/a", "none", "unknown", "."}


def is_missing(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip().lower() in MISSING_TOKENS


@st.cache_data
def load_pedigree(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Try comma first, fall back to whitespace delimiter. Read everything as string."""
    text = file_bytes.decode("utf-8-sig", errors="replace")

    # Try comma-delimited
    try:
        df = pd.read_csv(io.StringIO(text), sep=",", dtype=str, engine="python")
        if df.shape[1] >= 3:
            return df
    except Exception:
        pass

    # Fall back to whitespace-delimited (any amount of space/tab)
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", dtype=str, engine="python")
    return df


def build_pedigree_dict(df: pd.DataFrame, animal_col: str, sire_col: str, dam_col: str):
    """Map animal_id -> (sire_id, dam_id), using None for missing parents."""
    ped = {}
    for _, row in df.iterrows():
        animal = str(row[animal_col]).strip()
        if not animal or is_missing(animal):
            continue
        sire = row[sire_col]
        dam = row[dam_col]
        sire = None if is_missing(sire) else str(sire).strip()
        dam = None if is_missing(dam) else str(dam).strip()
        ped[animal] = (sire, dam)
    return ped


def traverse_ancestors(focal_id: str, ped: dict, n_generations: int):
    """
    BFS through the pedigree up to n_generations back.
    Returns:
        nodes: dict node_key -> {"id": animal_id, "gen": int, "role": "sire"/"dam"/"focal", "is_dup": bool, "in_pedigree": bool}
        edges: list of (parent_node_key, child_node_key)
    Each node gets a unique key (not just the animal id) because the SAME id can
    legitimately appear more than once in the tree only if the pedigree is cyclic
    (which is exactly what we want to surface) OR if an animal is an ancestor via
    multiple paths (e.g. inbreeding) -- both are valid to show as separate nodes
    in a generation-expanded tree.
    """
    nodes = {}
    edges = []

    root_key = "0"
    in_ped = focal_id in ped
    nodes[root_key] = {
        "id": focal_id,
        "gen": 0,
        "role": "focal",
        "is_dup": False,
        "in_pedigree": in_ped,
    }

    frontier = [(root_key, focal_id, 0)]
    counter = 0

    while frontier:
        new_frontier = []
        for node_key, animal_id, gen in frontier:
            if gen >= n_generations:
                continue
            if animal_id not in ped:
                continue  # unknown animal, no further parents recorded
            sire, dam = ped[animal_id]

            for parent_id, role in ((sire, "sire"), (dam, "dam")):
                if parent_id is None:
                    continue
                counter += 1
                child_key = f"{gen+1}_{role}_{counter}"
                is_dup = parent_id == focal_id  # cycle detector
                nodes[child_key] = {
                    "id": parent_id,
                    "gen": gen + 1,
                    "role": role,
                    "is_dup": is_dup,
                    "in_pedigree": parent_id in ped,
                }
                edges.append((node_key, child_key))
                # Stop expanding further down a path that already cycled back
                # to the focal animal, to avoid infinite/huge loops.
                if not is_dup:
                    new_frontier.append((child_key, parent_id, gen + 1))
        frontier = new_frontier

    return nodes, edges


def render_graph(nodes: dict, edges: list, focal_id: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="line", nodesep="0.25", ranksep="0.9")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")
    dot.attr("edge", arrowhead="none")

    for key, info in nodes.items():
        label = info["id"]
        if not info["in_pedigree"] and info["role"] != "focal":
            label += "\n(unknown)"

        if info["role"] == "focal":
            fillcolor = "#FFD54F"  # amber for the focal animal
            penwidth = "2"
            color = "black"
        elif info["is_dup"]:
            fillcolor = "#E53935"  # red highlight = same ID as focal -> cycle
            penwidth = "3"
            color = "#7F0000"
            label += "\n⚠ CYCLE"
        elif info["role"] == "sire":
            fillcolor = "#90CAF9"  # blue-ish for sires
            penwidth = "1"
            color = "black"
        else:
            fillcolor = "#F48FB1"  # pink-ish for dams
            penwidth = "1"
            color = "black"

        dot.node(key, label=label, fillcolor=fillcolor, color=color, penwidth=penwidth)

    for parent_key, child_key in edges:
        # edge drawn from child (closer to focal) to parent visually;
        # we go focal -> ancestor direction since rankdir=LR
        dot.edge(parent_key, child_key)

    return dot


# ---------------------------------------------------------------------------
# Sidebar: file upload & column mapping
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Upload pedigree")
    uploaded = st.file_uploader(
        "Pedigree file (.csv, .txt, .ped, ...)",
        type=None,
        help="Comma- or whitespace-delimited file with at least 3 columns.",
    )

    if uploaded is not None:
        try:
            df = load_pedigree(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            df = None
    else:
        df = None

    if df is not None:
        st.success(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
        st.dataframe(df.head(5), use_container_width=True, height=160)

        st.header("2. Map columns")
        cols = list(df.columns)

        def guess(colnames, options):
            for o in options:
                for c in colnames:
                    if c.strip().lower() == o:
                        return c
            return colnames[0]

        animal_guess = guess(cols, ["animal", "id", "calf", "ego"])
        sire_guess = guess(cols, ["sire", "father", "sire_id"])
        dam_guess = guess(cols, ["dam", "mother", "dam_id"])

        animal_col = st.selectbox("Animal ID column", cols, index=cols.index(animal_guess))
        sire_col = st.selectbox(
            "Sire (father) column", cols, index=cols.index(sire_guess) if sire_guess in cols else 0
        )
        dam_col = st.selectbox(
            "Dam (mother) column", cols, index=cols.index(dam_guess) if dam_guess in cols else 0
        )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if df is None:
    st.info("👈 Upload a pedigree file to get started.")
    st.markdown(
        """
**Expected format** — a plain text file, comma- or space-delimited, with one row per animal:

```
animal,sire,dam
A1,S1,D1
A2,S1,D2
S1,S3,D3
...
```

Missing parents can be left blank, or use `0`/`NA`/`unknown`.
"""
    )
else:
    ped = build_pedigree_dict(df, animal_col, sire_col, dam_col)
    all_ids = sorted(ped.keys())

    c1, c2 = st.columns([2, 1])
    with c1:
        focal_id = st.selectbox(
            "Animal to visualise",
            options=all_ids,
            help="Choose the animal whose pedigree you want to trace back.",
        )
        # allow manual override / typing an ID not necessarily in the list (e.g. as offspring only)
        manual_id = st.text_input("...or type an animal ID directly (overrides selection above)", "")
        if manual_id.strip():
            focal_id = manual_id.strip()

    with c2:
        n_gen = st.number_input(
            "Number of generations (N)", min_value=1, max_value=20, value=4, step=1
        )

    if focal_id not in ped:
        st.warning(
            f"⚠ '{focal_id}' was not found as an **animal** (offspring) in the pedigree file — "
            "it may only appear as a sire/dam, so no further ancestors can be traced for it."
        )

    nodes, edges = traverse_ancestors(focal_id, ped, int(n_gen))

    n_dups = sum(1 for n in nodes.values() if n["is_dup"])
    if n_dups > 0:
        st.error(
            f"🔴 Found **{n_dups}** ancestor node(s) sharing the same ID as the focal animal "
            f"'{focal_id}' — this indicates a **cyclic pedigree** (the animal is its own ancestor). "
            "These are highlighted in red below."
        )
    else:
        st.success("No cycles detected within the traced generations. ✅")

    st.subheader(f"Pedigree of '{focal_id}' — {n_gen} generation(s) back")

    graph = render_graph(nodes, edges, focal_id)
    st.graphviz_chart(graph, use_container_width=True)

    # Download button for the rendered image
    try:
        png_bytes = graph.pipe(format="png")
        st.download_button(
            "Download pedigree as PNG",
            data=png_bytes,
            file_name=f"pedigree_{focal_id}_{n_gen}gen.png",
            mime="image/png",
        )
    except Exception:
        pass

    with st.expander("Show traced nodes as a table"):
        table = pd.DataFrame(
            [
                {
                    "generation": v["gen"],
                    "role": v["role"],
                    "id": v["id"],
                    "in_pedigree_file": v["in_pedigree"],
                    "matches_focal_id": v["is_dup"],
                }
                for v in nodes.values()
            ]
        ).sort_values(["generation", "role"])
        st.dataframe(table, use_container_width=True, height=300)

    st.caption(
        "Legend — 🟡 focal animal · 🔵 sire · 🩷 dam · 🔴 ancestor with the same ID as the focal animal (cycle)."
    )
