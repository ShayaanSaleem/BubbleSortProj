import gradio as gr
import matplotlib.pyplot as plt
import time
import random
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

# =========================
# Global constants
# =========================

MAX_STUDENTS = 100

PALETTE = [
    "#ff6b6b",  # soft red
    "#4dabf7",  # sky blue
    "#51cf66",  # fresh green
    "#ffd43b",  # warm yellow
    "#b197fc",  # lavender purple
]

STEP_DELAY = 0.03   # delay after a single "Next step"
RUN_DELAY = 0.06    # delay between frames in "Run to end"

CANCEL_FLAG = False  # used to stop long animations

# =========================
# Gradio CSS / theme
# =========================

custom_css = """
body {
    background: #fff3e0;
}
.gradio-container {
    background: radial-gradient(circle at top left, #fff9f0 0, #ffe8d5 40%, #ffd7b8 100%);
}
/* General text colour */
.gradio-container label,
.gradio-container .gr-markdown,
.gradio-container .gr-markdown *,
.gradio-container .prose,
.gradio-container .prose * {
    color: #2b2b2b !important;
}
/* Inline code in markdown (e.g. `145, 162, 177`) */
.gradio-container .gr-markdown code,
.gradio-container .prose code {
    background: #ffe6bf !important;
    color: #2b2b2b !important;
    border-radius: 4px !important;
    padding: 0 3px !important;
}
/* Card-style boxes */
.gr-box {
    background: #fffdf7;
    border-radius: 14px;
    border: 1px solid #f1c9a5;
}
/* Buttons */
button, .gr-button {
    border-radius: 999px !important;
    border: 1px solid #f5b26b !important;
    background: #ffc46b !important;
    color: #4a2c16 !important;
    font-weight: 600 !important;
}
button:hover, .gr-button:hover {
    background: #ffb347 !important;
}
/* Slider track */
input[type="range"] {
    accent-color: #ffb347;
}
/* Number inputs for heights */
input[type="number"] {
    background: #fffdf7;
    border-radius: 8px;
    border: 1px solid #f1c9a5;
    color: #2b2b2b;
}
/* Explanation panel */
.gr-markdown {
    background: #fffdf7;
    border-radius: 14px;
    border: 1px solid #f1c9a5;
    padding: 12px;
}
"""

# =========================
# Helper functions
# =========================

def cm_to_feet_inches(cm_val: int) -> str:
    """Convert centimetres to a string like 5'11"."""
    total_inches = int(round(cm_val / 2.54))
    feet = total_inches // 12
    inches = total_inches % 12
    return f"{feet}'{inches}\""


def draw_characters(heights, colors=None, current_j=None, sorted_tail=0, initialized=False):
    """
    Draw the wall + all people, with heights mapped to the wall lines.
    """
    plt.close("all")

    n = len(heights)
    if n == 0:
        n = 6
        heights = [0] * n

    fig_width = min(18, max(10, n * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    fig.patch.set_facecolor("#fff9f0")
    ax.set_facecolor("#fffdf7")

    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylim(-0.35, 1.6)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    # ----- height wall -----
    num_lines = 11          # 140, 150, ..., 240
    cm_start = 140
    step_cm = 10
    cm_end = cm_start + step_cm * (num_lines - 1)

    y_bottom = 0.25         # y for 140 cm
    y_top = 1.35            # y for 240 cm

    def cm_to_y(cm_val: float) -> float:
        """Map a cm value to the vertical range of the wall."""
        cm_clamped = max(cm_start, min(cm_end, cm_val))
        t = (cm_clamped - cm_start) / (cm_end - cm_start)
        return y_bottom + t * (y_top - y_bottom)

    ys = [
        y_bottom + (y_top - y_bottom) * i / (num_lines - 1)
        for i in range(num_lines)
    ]

    for i, y in enumerate(ys):
        cm_val = cm_start + step_cm * i
        base_bold = (i % 2 == 0)
        logical_bold = (cm_val % 20 == 0)
        bold = base_bold or logical_bold

        line_width = 2.4 if bold else 1.0
        font_weight = "bold" if bold else "normal"
        font_size = 9 if bold else 8

        ax.hlines(
            y,
            -0.6,
            n - 0.1,
            colors="#3b3b3b",
            linewidth=line_width,
            zorder=1,
        )

        ax.text(
            -0.75,
            y,
            f"{cm_val} cm",
            ha="right",
            va="center",
            fontsize=font_size,
            color="#3b3b3b",
            fontweight=font_weight,
            zorder=3,
        )

        ax.text(
            n + 0.05,
            y,
            cm_to_feet_inches(cm_val),
            ha="left",
            va="center",
            fontsize=font_size,
            color="#3b3b3b",
            fontweight=font_weight,
            zorder=3,
        )

    # ----- colours for people -----
    if not initialized:
        base_colors = ["#d0d0d0"] * n
    else:
        if colors is None or len(colors) != n:
            base_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
        else:
            base_colors = list(colors)

    highlight_indices = set()
    if initialized and current_j is not None:
        for idx in (current_j, current_j + 1):
            if 0 <= idx < n:
                highlight_indices.add(idx)

    # ----- draw people -----
    if n <= 20:
        base_body_width = 0.5
    elif n <= 60:
        base_body_width = 0.35
    else:
        base_body_width = 0.2

    feet_y = -0.20
    max_body_height = cm_to_y(cm_end) - feet_y

    for i in range(n):
        color = base_colors[i]

        if initialized and heights[i] is not None:
            height_cm = heights[i]
            head_top_y = cm_to_y(height_cm)
            total_body_height = head_top_y - feet_y
            if total_body_height <= 0:
                total_body_height = 0.3
        else:
            total_body_height = 0.35
            head_top_y = feet_y + total_body_height

        # height split into legs / torso / head
        leg_height = total_body_height * 0.4
        torso_height = total_body_height * 0.4
        head_height = total_body_height * 0.2

        # shorter people are also a bit slimmer
        if initialized and heights[i] is not None and max_body_height > 0:
            height_ratio = total_body_height / max_body_height
            height_ratio = max(0.0, min(1.0, height_ratio))
            width_scale = 0.5 + 0.5 * height_ratio
            body_width = base_body_width * width_scale
        else:
            body_width = base_body_width * 0.7

        leg_width = body_width * 0.22

        leg_bottom = feet_y
        leg_top = leg_bottom + leg_height
        torso_bottom = leg_top
        torso_top = torso_bottom + torso_height
        head_bottom = torso_top
        head_top = head_bottom + head_height

        head_radius = head_height / 2
        head_center_y = head_bottom + head_radius

        edge_color = color
        edge_width = 2.0 if i in highlight_indices else 0.9
        z = 2

        torso_left = i - body_width / 2
        torso = FancyBboxPatch(
            (torso_left, torso_bottom),
            body_width,
            torso_height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            zorder=z,
        )
        ax.add_patch(torso)

        left_leg_x = i - body_width * 0.28
        right_leg_x = i + body_width * 0.06
        left_leg = Rectangle(
            (left_leg_x, leg_bottom),
            leg_width,
            leg_height,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            zorder=z,
        )
        right_leg = Rectangle(
            (right_leg_x, leg_bottom),
            leg_width,
            leg_height,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            zorder=z,
        )
        ax.add_patch(left_leg)
        ax.add_patch(right_leg)

        head = Circle(
            (i, head_center_y),
            head_radius,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            zorder=z,
        )
        ax.add_patch(head)

        if initialized and heights[i] is not None:
            label_y = cm_to_y(heights[i])
            ax.text(
                i,
                label_y + 0.03,
                f"{heights[i]}",
                ha="center",
                va="bottom",
                fontsize=8 if n > 40 else 10,
                color="#3b3b3b",
                fontweight="bold",
                zorder=4,
            )

        ax.text(
            i,
            feet_y - 0.03,
            f"Person {i+1}",
            ha="center",
            va="top",
            fontsize=8,
            color="#3b3b3b",
            zorder=4,
        )

    ax.set_title(
        "Bubble Height Sorter",
        fontsize=16,
        pad=12,
        color="#3b3b3b",
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def make_explanation(state):
    """Return the Markdown explanation text for the current sort state."""
    base = (
        "### Bubble Height Sorter\n\n"
        "- We’re using **Bubble Sort** to line students up from **shortest to tallest**.\n"
        "- Bubble Sort looks at **neighbours from left to right**.\n"
        "- If the person on the left is **taller** than the person on the right, they **swap places**.\n"
        "- After each full pass, the tallest person so far has \"**bubbled**\" to the right end.\n"
        "- The algorithm stops when it finishes a pass with **no swaps**.\n\n"
    )

    if state is None or not state.get("initialized", False):
        return base + (
            "Right now there is no active sorting step.\n\n"
            "**How to use this tool:**\n"
            "1. Choose the number of students with the slider.\n"
            "2. Type heights (in cm) into the boxes for Person 1, Person 2, etc., or click **Randomize heights**.\n"
            "3. Click **Set heights** to place them in front of the wall.\n"
            "4. Use **Next step** to watch Bubble Sort one comparison at a time, "
            "or **Run to end** to watch the full sort.\n"
        )

    heights = state["heights"]
    n = state["n"]
    i = state["pass_index"]
    j = state["inner_index"]
    sorted_tail = state["sorted_tail"]
    done = state["done"]

    heights_str = ", ".join(str(h) for h in heights)

    if done:
        return base + (
            f"**Sorting is finished.**\n\n"
            f"- The students are now in order from **shortest to tallest**.\n"
            f"- Heights from left to right: `{heights_str}`.\n\n"
            "In Bubble Sort terms:\n"
            f"- We completed {i} full pass(es), and the last pass had **no swaps**, "
            "so the algorithm knows the list is sorted.\n"
        )

    pass_num = i + 1
    total_passes = max(1, n - 1)

    if j < n - 1 - i:
        idx_left = j
        idx_right = j + 1
        h_left = heights[idx_left]
        h_right = heights[idx_right]
        person_left = idx_left + 1
        person_right = idx_right + 1

        return base + (
            f"**Current status:**\n\n"
            f"- We are on **pass {pass_num}** out of at most **{total_passes}**.\n"
            f"- Comparing **Person {person_left}** (height `{h_left} cm`) "
            f"and **Person {person_right}** (height `{h_right} cm`).\n"
            "- If the person on the left is taller, they swap. If not, they stay in place.\n\n"
            "Bubble Sort perspective:\n"
            f"- This comparison is at inner index **j = {j}**, pass **i = {i}**.\n"
            f"- The last **{sorted_tail}** people on the right are already in their final place.\n\n"
            f"Heights left to right now: `{heights_str}`.\n"
        )
    else:
        return base + (
            f"**Completed pass {pass_num}.**\n\n"
            f"- After this pass, the tallest unsorted student has moved into position on the **right**.\n"
            f"- Now there are **{sorted_tail}** student(s) at the right end that are guaranteed sorted.\n\n"
            "Bubble Sort perspective:\n"
            f"- We finished scanning neighbours up to index `{n - 1 - i}`.\n"
            "- The next pass will scan a slightly shorter portion of the line, "
            "because the rightmost students are already in place.\n\n"
            f"Heights left → right right now: `{heights_str}`.\n"
        )


INITIAL_EXPLANATION = make_explanation(None)

# =========================
# Bubble Sort state / logic
# =========================

def init_state_from_numbers(num_students, *height_values):
    """Read slider + inputs and create the initial sort state."""
    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    selected = list(height_values[:n])

    if any(h is None for h in selected):
        fig = draw_characters([0] * n, initialized=False)
        explanation = make_explanation(None)
        return fig, None, explanation

    try:
        heights = [int(h) for h in selected]
    except ValueError:
        fig = draw_characters([0] * n, initialized=False)
        explanation = make_explanation(None)
        return fig, None, explanation

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    state = {
        "heights": heights.copy(),
        "colors": colors,
        "n": n,
        "pass_index": 0,
        "inner_index": 0,
        "sorted_tail": 0,
        "swapped_in_pass": False,
        "done": False,
        "initialized": True,
        "num_students": n,
    }

    fig = draw_characters(heights, colors=colors, current_j=0, sorted_tail=0, initialized=True)
    explanation = make_explanation(state)
    return fig, state, explanation


def _bubble_sort_step_once(state):
    """Perform one Bubble Sort step and update the state."""
    if state is None or not state.get("initialized", False):
        fig = draw_characters([], initialized=False)
        explanation = make_explanation(state)
        return fig, state, explanation

    heights = state["heights"]
    colors = state.get("colors", [PALETTE[i % len(PALETTE)] for i in range(state["n"])])
    n = state["n"]
    i = state["pass_index"]
    j = state["inner_index"]
    sorted_tail = state["sorted_tail"]
    swapped_in_pass = state["swapped_in_pass"]
    done = state["done"]

    if done:
        fig = draw_characters(
            heights, colors=colors, current_j=None, sorted_tail=sorted_tail, initialized=True
        )
        explanation = make_explanation(state)
        return fig, state, explanation

    if j < n - 1 - i:
        if heights[j] > heights[j + 1]:
            heights[j], heights[j + 1] = heights[j + 1], heights[j]
            colors[j], colors[j + 1] = colors[j + 1], colors[j]
            swapped_in_pass = True

        j += 1
        state.update(
            {
                "heights": heights,
                "colors": colors,
                "inner_index": j,
                "swapped_in_pass": swapped_in_pass,
            }
        )

        highlight_j = j if j < n - 1 - i else None
        fig = draw_characters(
            heights,
            colors=colors,
            current_j=highlight_j,
            sorted_tail=sorted_tail,
            initialized=True,
        )
        explanation = make_explanation(state)
        return fig, state, explanation

    else:
        sorted_tail = i + 1

        if not swapped_in_pass:
            done = True
            sorted_tail = n
        else:
            i += 1
            j = 0
            swapped_in_pass = False
            if i >= n - 1:
                done = True
                sorted_tail = n

        state.update(
            {
                "heights": heights,
                "colors": colors,
                "pass_index": i,
                "inner_index": j,
                "sorted_tail": sorted_tail,
                "swapped_in_pass": swapped_in_pass,
                "done": done,
            }
        )

        highlight_j = j if (not done and j < n - 1 - i) else None
        fig = draw_characters(
            heights,
            colors=colors,
            current_j=highlight_j,
            sorted_tail=sorted_tail,
            initialized=True,
        )
        explanation = make_explanation(state)
        return fig, state, explanation


def bubble_sort_step_stream(state):
    """Generator used by the Next step button."""
    fig, new_state, explanation = _bubble_sort_step_once(state)
    time.sleep(STEP_DELAY)
    yield fig, new_state, explanation


def run_to_end(state):
    """Run Bubble Sort until finished or until cancelled by Reset."""
    global CANCEL_FLAG
    CANCEL_FLAG = False

    if state is None or not state.get("initialized", False):
        fig = draw_characters([], initialized=False)
        explanation = make_explanation(state)
        yield fig, state, explanation
        return

    heights = state["heights"]
    colors = state.get("colors", [PALETTE[i % len(PALETTE)] for i in range(state["n"])])

    fig = draw_characters(
        heights,
        colors=colors,
        current_j=state["inner_index"] if not state["done"] else None,
        sorted_tail=state["sorted_tail"],
        initialized=True,
    )
    explanation = make_explanation(state)
    yield fig, state, explanation
    time.sleep(RUN_DELAY)

    while not state["done"] and not CANCEL_FLAG:
        fig, state, explanation = _bubble_sort_step_once(state)
        yield fig, state, explanation
        time.sleep(RUN_DELAY)


def reset_app(num_students):
    """
    Reset for the given number of students:
    stop animation, clear inputs, show grey placeholders.
    """
    global CANCEL_FLAG
    CANCEL_FLAG = True

    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    fig = draw_characters([0] * n, initialized=False)
    explanation = make_explanation(None)

    input_updates = []
    for i in range(MAX_STUDENTS):
        if i < n:
            input_updates.append(gr.update(value=None, visible=True))
        else:
            input_updates.append(gr.update(value=None, visible=False))

    return (fig, None, explanation, *input_updates)


def init_app(num_students):
    """Initial load behaves like a reset."""
    return reset_app(num_students)


def randomize_heights(num_students, *current_values):
    """Random heights, update inputs, and initialise Bubble Sort state."""
    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    heights = [random.randint(140, 200) for _ in range(n)]
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    state = {
        "heights": heights.copy(),
        "colors": colors,
        "n": n,
        "pass_index": 0,
        "inner_index": 0,
        "sorted_tail": 0,
        "swapped_in_pass": False,
        "done": False,
        "initialized": True,
        "num_students": n,
    }

    fig = draw_characters(heights, colors=colors, current_j=0, sorted_tail=0, initialized=True)
    explanation = make_explanation(state)

    updates = []
    for i in range(MAX_STUDENTS):
        if i < n:
            updates.append(gr.update(value=heights[i], visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))

    return (fig, state, explanation, *updates)


# =========================
# Gradio UI wiring
# =========================

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="blue",
    neutral_hue="gray",
)

with gr.Blocks(theme=theme, css=custom_css) as demo:
    with gr.Row():
        # left side: main UI
        with gr.Column(scale=3):
            num_students_input = gr.Slider(
                minimum=2,
                maximum=MAX_STUDENTS,
                value=6,
                step=1,
                label="Number of students in the line",
            )

            plot_output = gr.Plot(
                label="Height line-up visualization",
            )

            with gr.Row():
                set_button = gr.Button("Set heights")
                random_button = gr.Button("Randomize heights")
                step_button = gr.Button("Next step")
                run_button = gr.Button("Run to end")
                reset_button = gr.Button("Reset")

            heights_inputs = []
            rows = 10
            cols = 10
            for r in range(rows):
                with gr.Row():
                    for c in range(cols):
                        idx = r * cols + c
                        num = gr.Number(
                            label=f"P{idx+1}",
                            precision=0,
                            visible=(idx < 6),
                            scale=1,
                            min_width=70,
                        )
                        heights_inputs.append(num)

        # right side: explanation
        with gr.Column(scale=2):
            explanation_md = gr.Markdown(
                value=INITIAL_EXPLANATION,
                label="What Bubble Sort is doing",
            )

    algo_state = gr.State()

    set_button.click(
        fn=init_state_from_numbers,
        inputs=[num_students_input] + heights_inputs,
        outputs=[plot_output, algo_state, explanation_md],
    )

    random_button.click(
        fn=randomize_heights,
        inputs=[num_students_input] + heights_inputs,
        outputs=[plot_output, algo_state, explanation_md] + heights_inputs,
    )

    step_button.click(
        fn=bubble_sort_step_stream,
        inputs=algo_state,
        outputs=[plot_output, algo_state, explanation_md],
    )

    run_button.click(
        fn=run_to_end,
        inputs=algo_state,
        outputs=[plot_output, algo_state, explanation_md],
    )

    reset_button.click(
        fn=reset_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state, explanation_md] + heights_inputs,
    )

    num_students_input.change(
        fn=reset_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state, explanation_md] + heights_inputs,
    )

    demo.load(
        fn=init_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state, explanation_md] + heights_inputs,
    )


if __name__ == "__main__":
    demo.launch()
