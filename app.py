import gradio as gr
import matplotlib.pyplot as plt
import time
import random
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

# --- Visualization constants --------------------------------------------------

MAX_STUDENTS = 100         # up to 100 students
PALETTE = [
    "#ff0000",  # red
    "#0000ff",  # blue
    "#008000",  # green
    "#ffd700",  # yellow (gold)
    "#800080",  # purple
]


# --- Helpers -------------------------------------------------------------------

def cm_to_feet_inches(cm_val: int) -> str:
    """Convert a height in cm to a string like 5'11"."""
    total_inches = int(round(cm_val / 2.54))
    feet = total_inches // 12
    inches = total_inches % 12
    return f"{feet}'{inches}\""


def draw_characters(
    heights,
    colors=None,
    current_j=None,
    sorted_tail=0,  # kept for compatibility; not used for coloring now
    initialized=False,
):
    """
    Draw the line of "students" in front of a police-style height wall.
    Left: cm. Right: feet + inches.
    Person height is mapped directly to the wall marks (140–240 cm).
    Colors come from 'colors' list (cycled red/blue/green/yellow/purple).
    """
    plt.close("all")

    n = len(heights)
    if n == 0:
        n = 6
        heights = [0] * n

    fig_width = min(18, max(10, n * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    fig.patch.set_facecolor("#e8e8e8")   # outer background
    ax.set_facecolor("#f5f0e7")          # wall colour

    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylim(-0.25, 1.6)  # little extra space below for feet
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    # -----------------------------
    # Height wall definition
    # -----------------------------
    num_lines = 11            # 140, 150, ..., 240
    cm_start = 140
    step_cm = 10
    cm_end = cm_start + step_cm * (num_lines - 1)  # 240

    y_bottom = 0.25           # y position for 140 cm line
    y_top = 1.35              # y position for 240 cm line

    def cm_to_y(cm_val: float) -> float:
        """Map a cm height onto [y_bottom, y_top]."""
        cm_clamped = max(cm_start, min(cm_end, cm_val))
        t = (cm_clamped - cm_start) / (cm_end - cm_start)
        return y_bottom + t * (y_top - y_bottom)

    ys = [
        y_bottom + (y_top - y_bottom) * i / (num_lines - 1)
        for i in range(num_lines)
    ]

    # draw wall lines + labels (zorder=1 so people can go in front)
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
            colors="black",
            linewidth=line_width,
            zorder=1,
        )

        left_label = f"{cm_val} cm"
        ax.text(
            -0.75,
            y,
            left_label,
            ha="right",
            va="center",
            fontsize=font_size,
            color="black",
            fontweight=font_weight,
            zorder=3,
        )

        right_label = cm_to_feet_inches(cm_val)
        ax.text(
            n + 0.05,
            y,
            right_label,
            ha="left",
            va="center",
            fontsize=font_size,
            color="black",
            fontweight=font_weight,
            zorder=3,
        )

    # -----------------------------
    # Colors (per person, following the palette)
    # -----------------------------
    if not initialized:
        base_colors = ["#cccccc"] * n
    else:
        if colors is None or len(colors) != n:
            base_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
        else:
            base_colors = list(colors)

    # Which indices are currently being compared
    highlight_indices = set()
    if initialized and current_j is not None:
        for idx in (current_j, current_j + 1):
            if 0 <= idx < n:
                highlight_indices.add(idx)

    # -----------------------------
    # Draw each person (no arms)
    # -----------------------------
    if n <= 20:
        base_body_width = 0.5
    elif n <= 60:
        base_body_width = 0.35
    else:
        base_body_width = 0.2

    feet_y = -0.10  # floor line

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

        # proportions
        leg_height = total_body_height * 0.3
        torso_height = total_body_height * 0.4
        head_height = total_body_height * 0.3

        body_width = base_body_width
        leg_width = body_width * 0.22

        # vertical layout
        leg_bottom = feet_y
        leg_top = leg_bottom + leg_height
        torso_bottom = leg_top
        torso_top = torso_bottom + torso_height
        head_bottom = torso_top

        if initialized and heights[i] is not None:
            head_top = head_top_y
            head_height = max(0.05, head_top - head_bottom)
        else:
            head_top = head_bottom + head_height

        head_radius = head_height / 2
        head_center_y = head_bottom + head_radius

        # outline style: same colour as body; highlighted ones just thicker
        edge_color = color
        edge_width = 2.0 if i in highlight_indices else 0.9

        z = 2  # zorder for bodies so they sit on top of lines

        # torso
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

        # legs
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

        # head
        head = Circle(
            (i, head_center_y),
            head_radius,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            zorder=z,
        )
        ax.add_patch(head)

        # numeric height label
        if initialized and heights[i] is not None:
            ax.text(
                i,
                head_top_y + 0.03,
                f"{heights[i]}",
                ha="center",
                va="bottom",
                fontsize=8 if n > 40 else 10,
                color="black",
                fontweight="bold",
                zorder=4,
            )

        # index at feet
        ax.text(
            i,
            feet_y - 0.03,
            f"{i}",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            zorder=4,
        )

    ax.set_title(
        "Bubble Height Sorter",
        fontsize=16,
        pad=12,
        color="black",
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# --- Bubble sort state + step logic -------------------------------------------

def init_state_from_numbers(num_students, *height_values):
    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    selected = list(height_values[:n])

    if any(h is None for h in selected):
        fig = draw_characters([0] * n, initialized=False)
        return fig, None

    try:
        heights = [int(h) for h in selected]
    except ValueError:
        fig = draw_characters([0] * n, initialized=False)
        return fig, None

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
    return fig, state


def _bubble_sort_step_once(state):
    if state is None or not state.get("initialized", False):
        fig = draw_characters([], initialized=False)
        return fig, state

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
        return fig, state

    if j < n - 1 - i:
        if heights[j] > heights[j + 1]:
            # swap heights AND colors so the same person keeps their colour
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
        return fig, state

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
        return fig, state


def bubble_sort_step_stream(state):
    fig, new_state = _bubble_sort_step_once(state)
    time.sleep(0.05)
    yield fig, new_state


def run_to_end(state):
    if state is None or not state.get("initialized", False):
        fig = draw_characters([], initialized=False)
        yield fig, state
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
    yield fig, state
    time.sleep(0.15)

    while not state["done"]:
        fig, state = _bubble_sort_step_once(state)
        yield fig, state
        time.sleep(0.15)


def reset_app(num_students):
    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    fig = draw_characters([0] * n, initialized=False)
    return fig, None


def update_inputs_visibility(num_students):
    try:
        n = int(num_students)
    except Exception:
        n = 6
    n = max(2, min(MAX_STUDENTS, n))

    updates = []
    for i in range(MAX_STUDENTS):
        if i < n:
            updates.append(gr.update(visible=True))
        else:
            updates.append(gr.update(visible=False, value=None))
    return updates


def init_app(num_students):
    fig, state = reset_app(num_students)
    vis_updates = update_inputs_visibility(num_students)
    return (fig, state, *vis_updates)


def randomize_heights(num_students, *current_values):
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

    updates = []
    for i in range(MAX_STUDENTS):
        if i < n:
            updates.append(gr.update(value=heights[i], visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))

    return (fig, state, *updates)


# --- Gradio interface ---------------------------------------------------------


with gr.Blocks() as demo:
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

    heights_inputs = []
    rows = 10
    cols = 10
    for r in range(rows):
        with gr.Row():
            for c in range(cols):
                idx = r * cols + c
                num = gr.Number(label=str(idx), precision=0, visible=(idx < 6))
                heights_inputs.append(num)

    with gr.Row():
        set_button = gr.Button("Set heights")
        random_button = gr.Button("Randomize heights")
        step_button = gr.Button("Next step")
        run_button = gr.Button("Run to end")
        reset_button = gr.Button("Reset")

    algo_state = gr.State()

    set_button.click(
        fn=init_state_from_numbers,
        inputs=[num_students_input] + heights_inputs,
        outputs=[plot_output, algo_state],
    )

    random_button.click(
        fn=randomize_heights,
        inputs=[num_students_input] + heights_inputs,
        outputs=[plot_output, algo_state] + heights_inputs,
    )

    step_button.click(
        fn=bubble_sort_step_stream,
        inputs=algo_state,
        outputs=[plot_output, algo_state],
    )

    run_button.click(
        fn=run_to_end,
        inputs=algo_state,
        outputs=[plot_output, algo_state],
    )

    reset_button.click(
        fn=reset_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state],
    )

    num_students_input.change(
        fn=reset_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state],
    )

    num_students_input.change(
        fn=update_inputs_visibility,
        inputs=num_students_input,
        outputs=heights_inputs,
    )

    demo.load(
        fn=init_app,
        inputs=num_students_input,
        outputs=[plot_output, algo_state] + heights_inputs,
    )


if __name__ == "__main__":
    demo.launch()
