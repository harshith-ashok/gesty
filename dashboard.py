import gradio as gr
import json
from pathlib import Path

STATE_FILE = Path(__file__).with_name("device_state.json")


def get_status():
    try:
        if not STATE_FILE.exists():
            return "# No device state file found"

        with open(STATE_FILE, "r") as f:
            data = json.load(f)

        devices = data.get("devices", {})
        output = []

        for name, device in devices.items():
            status = "ON" if device.get("status", 0) == 1 else "OFF"

            section = f"## {name}\n\n"
            section += f"**Status:** {status}\n"

            if name == "Ceiling Fan":
                value = device.get("value", 0)
                section += f"\n**Speed:** {value}%"

            if name == "Accent Light":
                color = device.get("color", "none")
                section += f"\n**Color:** {str(color).upper()}"

            output.append(section)

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"# Error\n\n```{e}```"


with gr.Blocks(title="Smart Home Dashboard") as demo:
    gr.Markdown("# Smart Home Dashboard")
    gr.Markdown("Live device status from `device_state.json`.")

    status_box = gr.Markdown()

    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(
        fn=get_status,
        outputs=status_box
    )

    gr.Timer(2).tick(
        fn=get_status,
        outputs=status_box
    )

    demo.load(
        fn=get_status,
        outputs=status_box
    )


if __name__ == "__main__":
    demo.launch()
