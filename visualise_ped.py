import streamlit as st
import pandas as pd
import graphviz
import io

st.set_page_config(page_title="Pedigree Viewer", layout="wide")

st.title("🐄 Pedigree Visualiser")
st.caption(
    "Upload a pedigree file (comma- or space-delimited, 3 columns: animal, sire, dam), "
    "pick an animal, and explore its ancestry for N generations. "
    "Ancestors sharing the focal animal's ID are highlighted in yellow — a sign of a cyclic pedigree."
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


@st.cache_data
def build_pedigree_dict(
    df: pd.DataFrame,
    animal_col: str,
    sire_col: str,
    dam_col: str,
    birth_col: str | None = None,
):
    """Map animal_id -> (sire_id, dam_id, birth_date), using None for missing values.

    birth_date is optional (None for every animal if birth_col is not given).
    Vectorized (no iterrows) so this stays fast on pedigrees with hundreds of
    thousands of rows; also cached so it only reruns when the file or column
    mapping actually changes, not on every widget interaction.
    """
    use_cols = [animal_col, sire_col, dam_col] + ([birth_col] if birth_col else [])
    sub = df[use_cols].astype(str)
    new_names = ["animal", "sire", "dam"] + (["birth"] if birth_col else [])
    sub.columns = new_names
    sub["animal"] = sub["animal"].str.strip()

    def clean_parent_col(s: pd.Series) -> pd.Series:
        s = s.str.strip()
        is_miss = s.str.lower().isin(MISSING_TOKENS)
        out = s.astype(object)
        out[is_miss.values] = None
        return out

    sub["sire"] = clean_parent_col(sub["sire"])
    sub["dam"] = clean_parent_col(sub["dam"])
    if birth_col:
        sub["birth"] = clean_parent_col(sub["birth"])
    else:
        sub["birth"] = None

    animal_lower = sub["animal"].str.lower()
    sub = sub[~animal_lower.isin(MISSING_TOKENS)]
    # if an animal id appears more than once in the file, keep the last record
    sub = sub.drop_duplicates(subset="animal", keep="last")

    return dict(zip(sub["animal"], zip(sub["sire"], sub["dam"], sub["birth"])))


def traverse_ancestors(focal_id: str, ped: dict, n_generations: int):
    """
    BFS through the pedigree up to n_generations back.
    Returns:
        nodes: dict node_key -> {"id", "gen", "role", "is_dup", "in_pedigree", "birth"}
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
    root_birth = ped[focal_id][2] if in_ped else None
    nodes[root_key] = {
        "id": focal_id,
        "gen": 0,
        "role": "focal",
        "is_dup": False,
        "in_pedigree": in_ped,
        "birth": root_birth,
    }

    frontier = [(root_key, focal_id, 0)]
    counter = 0
    MAX_NODES = 5000  # safety cap: a fully-expanded binary tree hits this by gen ~12
    truncated = False

    while frontier:
        new_frontier = []
        for node_key, animal_id, gen in frontier:
            if gen >= n_generations:
                continue
            if animal_id not in ped:
                continue  # unknown animal, no further parents recorded
            sire, dam, _birth = ped[animal_id]

            for parent_id, role in ((sire, "sire"), (dam, "dam")):
                if parent_id is None:
                    continue
                if len(nodes) >= MAX_NODES:
                    truncated = True
                    break
                counter += 1
                child_key = f"{gen+1}_{role}_{counter}"
                is_dup = parent_id == focal_id  # cycle detector
                parent_birth = ped[parent_id][2] if parent_id in ped else None
                nodes[child_key] = {
                    "id": parent_id,
                    "gen": gen + 1,
                    "role": role,
                    "is_dup": is_dup,
                    "in_pedigree": parent_id in ped,
                    "birth": parent_birth,
                }
                edges.append((node_key, child_key))
                # Stop expanding further down a path that already cycled back
                # to the focal animal, to avoid infinite/huge loops.
                if not is_dup:
                    new_frontier.append((child_key, parent_id, gen + 1))
            if truncated:
                break
        if truncated:
            break
        frontier = new_frontier

    return nodes, edges, truncated


def build_pedigree_table_tree(focal_id: str, ped: dict, n_generations: int):
    """
    Build a strict binary tree (sire branch, dam branch) of ancestors for the
    focal animal, suitable for rendering as a nested table like a classic
    pedigree chart (columns = generations, cells span their descendants' rows).

    Returns (tree, truncated). tree is a nested dict:
        {"id": str, "birth": str|None, "role": "focal"/"sire"/"dam",
         "in_pedigree": bool, "is_dup": bool,
         "sire": <node or None>, "dam": <node or None>}
    Recursion stops at n_generations, at unknown animals, or the moment a
    node's id matches focal_id again (cycle) -- that node is still shown
    (flagged) but not expanded further, to keep a true cycle from looping
    forever. A strict binary tree can have up to 2**N leaf cells, so this is
    capped (truncated=True if hit) to keep the table renderable.
    """
    MAX_LEAVES = 4096  # 2**12; keeps the table from becoming unusably huge
    state = {"count": 1, "truncated": False}

    def expand(animal_id, role, gen):
        in_ped = animal_id in ped
        is_dup = (animal_id == focal_id) and gen > 0
        birth = ped[animal_id][2] if in_ped else None
        node = {
            "id": animal_id,
            "birth": birth,
            "role": role,
            "in_pedigree": in_ped,
            "is_dup": is_dup,
            "sire": None,
            "dam": None,
        }
        if gen >= n_generations or not in_ped or is_dup:
            return node
        if state["count"] * 2 > MAX_LEAVES:
            state["truncated"] = True
            return node
        sire_id, dam_id, _ = ped[animal_id]
        if sire_id is not None:
            state["count"] += 1
            node["sire"] = expand(sire_id, "sire", gen + 1)
        if dam_id is not None:
            state["count"] += 1
            node["dam"] = expand(dam_id, "dam", gen + 1)
        return node

    tree = expand(focal_id, "focal", 0)
    return tree, state["truncated"]


def render_pedigree_table(tree: dict, n_generations: int) -> str:
    """
    Render the binary ancestor tree as ONE nested HTML table, matching a classic
    pedigree chart: one column per generation (0 = focal animal .. N = most
    distant ancestors), each cell vertically spanning the rows of its own
    descendants further to the right. Sire's whole subtree occupies the top
    half of rows, dam's subtree the bottom half (recursively), just like a
    standard pedigree chart.
    """

    def cell_html(node) -> str:
        if node is None:
            return "<div class='ped-cell ped-empty'>&nbsp;</div>"
        classes = ["ped-cell"]
        if node["role"] == "focal":
            classes.append("ped-focal")
        elif node["is_dup"]:
            classes.append("ped-cycle")
        elif node["role"] == "sire":
            classes.append("ped-sire")
        else:
            classes.append("ped-dam")
        if not node["in_pedigree"] and node["role"] != "focal":
            classes.append("ped-leaf")

        name = node["id"]
        birth = f"<span class='ped-birth'>{node['birth']}</span>" if node.get("birth") else ""
        flag = "<span class='ped-flag'>⚠ CYCLE</span>" if node["is_dup"] else ""
        return (
            f"<div class='{' '.join(classes)}'>"
            f"<span class='ped-name'>{name}</span>{birth}{flag}"
            f"</div>"
        )

    # leaves(node, depth_remaining) = number of row-slots this node occupies
    # at the deepest rendered generation (i.e. size of its "row span").
    def leaves(node, depth_remaining):
        if node is None or depth_remaining <= 0:
            return 1
        if node.get("is_dup") or not node.get("in_pedigree", True):
            return 1
        s = leaves(node.get("sire"), depth_remaining - 1)
        d = leaves(node.get("dam"), depth_remaining - 1)
        return s + d

    total_rows = leaves(tree, n_generations)

    # columns[g] will collect (node, rowspan) tuples in top-to-bottom order
    # for generation g (0 = focal .. n_generations).
    columns = [[] for _ in range(n_generations + 1)]

    def walk(node, gen, depth_remaining):
        rs = leaves(node, depth_remaining)
        columns[gen].append((node, rs))
        if gen == n_generations:
            return
        if node is None or node.get("is_dup") or not node.get("in_pedigree", True):
            # pad remaining columns with a single empty cell of the same rowspan
            walk(None, gen + 1, depth_remaining - 1)
            return
        walk(node.get("sire"), gen + 1, depth_remaining - 1)
        walk(node.get("dam"), gen + 1, depth_remaining - 1)

    walk(tree, 0, n_generations)

    # Now emit row by row: for each of total_rows rows, for each column, emit
    # a <td rowspan=...> only when that column's next cell "starts" at this row.
    col_iters = [iter(col) for col in columns]
    remaining = [0] * (n_generations + 1)

    rows_html = []
    for r in range(total_rows):
        cells = []
        for c in range(n_generations + 1):
            if remaining[c] > 0:
                remaining[c] -= 1
                continue
            node, rs = next(col_iters[c])
            remaining[c] = rs - 1
            cells.append(f"<td rowspan='{rs}'>{cell_html(node)}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"<div class='ped-wrapper'><table class='ped-table'>{''.join(rows_html)}</table></div>"


PEDIGREE_TABLE_CSS = """
<style>
.ped-wrapper {
    overflow-x: auto;
    width: 100%;
    margin: 0.5rem 0 1rem 0;
}
table.ped-table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
table.ped-table td {
    border: 1px solid #D8DCE2;
    padding: 0;
    vertical-align: middle;
    min-width: 130px;
}
.ped-cell {
    padding: 6px 10px;
    line-height: 1.3;
    height: 100%;
    min-height: 36px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.ped-name {
    font-size: 13px;
    font-weight: 600;
    color: #1A1C1E;
    word-break: break-word;
}
.ped-birth {
    font-size: 11px;
    color: #6B7280;
    margin-top: 2px;
}
.ped-flag {
    font-size: 11px;
    font-weight: 700;
    color: #7F6F00;
    margin-top: 2px;
}
.ped-focal {
    background: #FFD54F;
}
.ped-sire {
    background: #E8F1FC;
}
.ped-dam {
    background: #FCE9F1;
}
.ped-cycle {
    background: #FFEB3B;
    border: 2px solid #7F6F00 !important;
}
.ped-leaf .ped-name {
    color: #9AA1AB;
    font-style: italic;
}
.ped-empty {
    background: #FAFAFA;
}
</style>
"""


def render_graph(nodes: dict, edges: list, focal_id: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="line", nodesep="0.25", ranksep="0.9")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")
    dot.attr("edge", arrowhead="none")

    for key, info in nodes.items():
        label = info["id"]
        if info.get("birth"):
            label += f"\n({info['birth']})"
        if not info["in_pedigree"] and info["role"] != "focal":
            label += "\n(unknown)"

        if info["role"] == "focal":
            fillcolor = "#FFD54F"  # amber for the focal animal
            penwidth = "2"
            color = "black"
        elif info["is_dup"]:
            fillcolor = "#FFEB3B"  # yellow highlight = same ID as focal -> cycle
            penwidth = "3"
            color = "#7F6F00"
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

        use_birth = st.checkbox("Include birth date column (optional)", value=False)
        if use_birth:
            birth_guess = guess(cols, ["birth", "birthdate", "birth_date", "dob", "born"])
            birth_col = st.selectbox(
                "Birth date column",
                cols,
                index=cols.index(birth_guess) if birth_guess in cols else 0,
            )
        else:
            birth_col = None

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

Missing parents can be left blank, or use `0`/`NA`/`unknown`. An optional 4th
column with each animal's birth date can also be mapped in the sidebar.
"""
    )
else:
    with st.spinner("Indexing pedigree..."):
        ped = build_pedigree_dict(df, animal_col, sire_col, dam_col, birth_col)

    n_animals = len(ped)
    st.caption(f"{n_animals:,} animals indexed.")

    c1, c2 = st.columns([2, 1])
    with c1:
        if "focal_id_input" not in st.session_state:
            st.session_state["focal_id_input"] = ""

        focal_id = st.text_input(
            "Animal ID to visualise",
            key="focal_id_input",
            help="Type the exact ID of the animal whose pedigree you want to trace back.",
        ).strip()

        # Instead of a dropdown with every ID (slow to ship to the browser at
        # scale, and not very usable either), offer a small live search: type
        # a few characters and see matching IDs, without ever rendering all
        # 200k options at once.
        with st.expander("🔍 Not sure of the exact ID? Search for it"):
            search_term = st.text_input(
                "Search animal IDs (substring match)", value="", key="id_search"
            ).strip().lower()
            if search_term:
                # cheap substring scan over keys; fine up to a few hundred k IDs
                matches = [aid for aid in ped.keys() if search_term in aid.lower()]
                MAX_MATCHES = 25
                if not matches:
                    st.write("No matching IDs.")
                else:
                    shown = sorted(matches)[:MAX_MATCHES]
                    suffix = f", showing first {MAX_MATCHES}" if len(matches) > MAX_MATCHES else ""
                    st.write(f"{len(matches):,} match(es){suffix}:")
                    picked = st.selectbox("Matching IDs", options=shown, key="id_search_pick")

                    def _use_picked_id():
                        st.session_state["focal_id_input"] = st.session_state["id_search_pick"]

                    st.button("Use this ID", on_click=_use_picked_id)

    with c2:
        n_gen = st.number_input(
            "Number of generations (N)", min_value=1, max_value=20, value=4, step=1
        )

    # Nothing is traced or rendered until the user explicitly asks for it —
    # avoids re-tracing/re-rendering on every keystroke or widget interaction.
    visualise_clicked = st.button("📊 Visualise pedigree", type="primary")

    if visualise_clicked:
        if not focal_id:
            st.warning("Please enter an animal ID first.")
            st.session_state["last_result"] = None
        else:
            st.session_state["last_result"] = {"focal_id": focal_id, "n_gen": int(n_gen)}

    result = st.session_state.get("last_result")

    if not result:
        st.info("👆 Enter an animal ID and click **Visualise pedigree** to get started.")
    else:
        focal_id = result["focal_id"]
        n_gen = result["n_gen"]

        if focal_id not in ped:
            st.warning(
                f"⚠ '{focal_id}' was not found as an **animal** (offspring) in the pedigree file — "
                "it may only appear as a sire/dam, so no further ancestors can be traced for it."
            )

        view_mode = st.radio(
            "Layout",
            ["Table (compact)", "Diagram (boxes & lines)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if view_mode == "Table (compact)":
            with st.spinner("Building pedigree table..."):
                tree, table_truncated = build_pedigree_table_tree(focal_id, ped, int(n_gen))

            if table_truncated:
                st.warning(
                    "⚠ This pedigree branch is very large (4,000+ ancestor cells) and the table "
                    "was truncated for performance. Try lowering N."
                )

            def count_dups(node):
                if node is None:
                    return 0
                n = 1 if node["is_dup"] else 0
                return n + count_dups(node.get("sire")) + count_dups(node.get("dam"))

            n_dups = count_dups(tree)
            if n_dups > 0:
                st.warning(
                    f"🟡 Found **{n_dups}** ancestor cell(s) sharing the same ID as the focal animal "
                    f"'{focal_id}' — this indicates a **cyclic pedigree** (the animal is its own ancestor). "
                    "These are highlighted in yellow below."
                )
            else:
                st.success("No cycles detected within the traced generations. ✅")

            st.subheader(f"Pedigree of '{focal_id}' — {n_gen} generation(s) back")

            table_html = render_pedigree_table(tree, int(n_gen))
            st.markdown(PEDIGREE_TABLE_CSS + table_html, unsafe_allow_html=True)

            st.caption(
                "Legend — 🟠 focal animal · 🔵 sire · 🩷 dam · 🟡 ancestor with the same ID as "
                "the focal animal (cycle) · grey = not found in the pedigree file (unknown)."
            )

        else:
            with st.spinner("Tracing ancestors..."):
                nodes, edges, truncated = traverse_ancestors(focal_id, ped, int(n_gen))

            if truncated:
                st.warning(
                    "⚠ The traced pedigree got very large (5,000+ nodes) and was truncated for "
                    "performance. This usually means deep inbreeding/many cycles combined with a "
                    "high N. Try lowering N or inspect the table below for what was captured."
                )

            n_dups = sum(1 for n in nodes.values() if n["is_dup"])
            if n_dups > 0:
                st.warning(
                    f"🟡 Found **{n_dups}** ancestor node(s) sharing the same ID as the focal animal "
                    f"'{focal_id}' — this indicates a **cyclic pedigree** (the animal is its own ancestor). "
                    "These are highlighted in yellow below."
                )
            else:
                st.success("No cycles detected within the traced generations. ✅")

            st.subheader(f"Pedigree of '{focal_id}' — {n_gen} generation(s) back")

            with st.spinner("Rendering graph..."):
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

            st.caption(
                "Legend — 🟠 focal animal · 🔵 sire · 🩷 dam · 🟡 ancestor with the same ID as the focal animal (cycle)."
            )

        with st.expander("Show traced ancestors as a plain data table"):
            with st.spinner("Tracing ancestors..."):
                flat_nodes, _, _ = traverse_ancestors(focal_id, ped, int(n_gen))
            flat_table = pd.DataFrame(
                [
                    {
                        "generation": v["gen"],
                        "role": v["role"],
                        "id": v["id"],
                        "birth_date": v.get("birth") or "",
                        "in_pedigree_file": v["in_pedigree"],
                        "matches_focal_id": v["is_dup"],
                    }
                    for v in flat_nodes.values()
                ]
            ).sort_values(["generation", "role"])
            st.dataframe(flat_table, use_container_width=True, height=300)
