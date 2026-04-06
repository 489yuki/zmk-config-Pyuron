#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


KEYCODE_MAP = {
    "AMPERSAND": "&",
    "ASTERISK": "*",
    "AT_SIGN": "@",
    "BACKSPACE": "BSPC",
    "COMMA": ",",
    "DOLLAR": "$",
    "DOT": ".",
    "DOWN_ARROW": "DOWN",
    "ENTER": "ENT",
    "EQUAL": "=",
    "ESCAPE": "ESC",
    "EXCLAMATION": "!",
    "GRAVE": "`",
    "HASH": "#",
    "K_MUTE": "MUTE",
    "LEFT_ALT": "LALT",
    "LEFT_BRACKET": "[",
    "LEFT_COMMAND": "LCMD",
    "LEFT_CONTROL": "LCTRL",
    "LEFT_SHIFT": "LSFT",
    "LEFT_ARROW": "LEFT",
    "MINUS": "-",
    "NON_US_BACKSLASH": "NUBS",
    "PERCENT": "%",
    "RIGHT_ALT": "RALT",
    "RIGHT_ARROW": "RIGHT",
    "RIGHT_BRACKET": "]",
    "RIGHT_SHIFT": "RSFT",
    "SEMICOLON": ";",
    "SLASH": "/",
    "SPACE": "SPC",
    "SQT": "'",
    "TAB": "TAB",
    "UP_ARROW": "UP",
}

MACRO_CONTROL_LABELS = {
    "&macro_pause_for_release": "wait-release",
    "&macro_press": "press",
    "&macro_release": "release",
    "&macro_tap": "tap",
}

PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN = 36
TITLE_GAP = 64
BOTTOM_MARGIN = 36


@dataclass(frozen=True)
class LayoutKey:
    x: float
    y: float
    width: float = 1.0
    height: float = 1.0


@dataclass(frozen=True)
class Layer:
    name: str
    bindings: list[str]


@dataclass(frozen=True)
class Combo:
    name: str
    binding: str
    positions: list[int]
    label: str | None = None


@dataclass(frozen=True)
class Macro:
    name: str
    label: str | None
    steps: list[str]


@dataclass(frozen=True)
class Behavior:
    name: str
    label: str | None
    bindings: list[str]


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def strip_line_comments(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(re.sub(r"//.*$", "", line))
    return "\n".join(cleaned_lines)


def extract_braced_block(text: str, open_brace_index: int) -> tuple[str, int]:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1 : index], index
    raise ValueError("Unbalanced braces in keymap file")


def extract_named_block(text: str, block_name: str) -> str:
    pattern = re.compile(rf"(?<![\w-]){re.escape(block_name)}\s*\{{")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find block '{block_name}'")
    return extract_braced_block(text, match.end() - 1)[0]


def extract_root_blocks(text: str) -> list[str]:
    roots = []
    for match in re.finditer(r"/\s*\{", text):
        roots.append(extract_braced_block(text, match.end() - 1)[0])
    return roots


def parse_child_nodes(block_text: str) -> list[tuple[str, str | None, str]]:
    nodes: list[tuple[str, str | None, str]] = []
    pattern = re.compile(
        r"""
        (?P<name>[A-Za-z_][\w-]*)
        (?:\s*:\s*(?P<label>[A-Za-z_][\w-]*))?
        \s*\{
        """,
        re.VERBOSE,
    )
    index = 0
    while index < len(block_text):
        match = pattern.search(block_text, index)
        if not match:
            break
        body, end_index = extract_braced_block(block_text, match.end() - 1)
        nodes.append((match.group("name"), match.group("label"), body))
        index = end_index + 1
    return nodes


def extract_property_value(block_text: str, prop_name: str) -> str | None:
    match = re.search(rf"(?<![\w-]){re.escape(prop_name)}\s*=", block_text)
    if not match:
        return None
    start = match.end()
    end = block_text.find(";", start)
    if end == -1:
        raise ValueError(f"Missing ';' after property '{prop_name}'")
    return block_text[start:end].strip()


def extract_string_property(block_text: str, prop_name: str) -> str | None:
    value = extract_property_value(block_text, prop_name)
    if value is None:
        return None
    match = re.fullmatch(r'"([^"]*)"', normalize_space(value))
    return match.group(1) if match else None


def extract_angle_groups(value: str | None, *, normalize: bool = True) -> list[str]:
    if not value:
        return []
    groups = re.findall(r"<([^>]*)>", value, flags=re.DOTALL)
    if normalize:
        return [normalize_space(group) for group in groups]
    return [group.strip("\n") for group in groups]


def split_layer_bindings(group: str) -> list[str]:
    bindings: list[str] = []
    for raw_line in group.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bindings.extend(part.strip() for part in re.split(r"\s{2,}", line) if part.strip())
    return bindings


def load_layout(layout_path: Path) -> list[LayoutKey]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    layouts = data.get("layouts", {})
    if not layouts:
        raise ValueError("Layout JSON does not contain any layouts")
    first_layout = next(iter(layouts.values()))
    raw_keys = first_layout.get("layout", [])
    if not raw_keys:
        raise ValueError("Layout JSON does not contain any key positions")
    return [
        LayoutKey(
            x=float(raw_key["x"]),
            y=float(raw_key["y"]),
            width=float(raw_key.get("w", 1.0)),
            height=float(raw_key.get("h", 1.0)),
        )
        for raw_key in raw_keys
    ]


def parse_layers(keymap_text: str) -> list[Layer]:
    keymap_block = extract_named_block(keymap_text, "keymap")
    layers: list[Layer] = []
    for name, _label, body in parse_child_nodes(keymap_block):
        binding_value = extract_property_value(body, "bindings")
        groups = extract_angle_groups(binding_value, normalize=False)
        if not groups:
            continue
        layers.append(Layer(name=name, bindings=split_layer_bindings(groups[0])))
    return layers


def parse_combos(keymap_text: str) -> list[Combo]:
    combo_block = extract_named_block(keymap_text, "combos")
    combos: list[Combo] = []
    for name, _label, body in parse_child_nodes(combo_block):
        binding_value = extract_property_value(body, "bindings")
        position_value = extract_property_value(body, "key-positions")
        binding_groups = extract_angle_groups(binding_value)
        position_groups = extract_angle_groups(position_value)
        if not binding_groups or not position_groups:
            continue
        combos.append(
            Combo(
                name=name,
                binding=binding_groups[0],
                positions=[int(value) for value in position_groups[0].split()],
                label=extract_string_property(body, "label"),
            )
        )
    return combos


def parse_macros(keymap_text: str) -> list[Macro]:
    macro_block = extract_named_block(keymap_text, "macros")
    macros: list[Macro] = []
    for name, _label, body in parse_child_nodes(macro_block):
        binding_value = extract_property_value(body, "bindings")
        groups = extract_angle_groups(binding_value)
        if not groups:
            continue
        macros.append(
            Macro(
                name=name,
                label=extract_string_property(body, "label"),
                steps=groups,
            )
        )
    return macros


def parse_custom_behaviors(keymap_text: str) -> list[Behavior]:
    behaviors: list[Behavior] = []
    for root_block in extract_root_blocks(keymap_text):
        for name, _label, body in parse_child_nodes(root_block):
            if name in {"behaviors", "combos", "keymap", "macros"}:
                continue
            binding_value = extract_property_value(body, "bindings")
            groups = extract_angle_groups(binding_value)
            if not groups:
                continue
            behaviors.append(
                Behavior(
                    name=name,
                    label=extract_string_property(body, "label"),
                    bindings=groups,
                )
            )
    return behaviors


def normalize_keycode(token: str) -> str:
    if token in KEYCODE_MAP:
        return KEYCODE_MAP[token]
    if token.startswith("NUMBER_"):
        return token.removeprefix("NUMBER_")
    if re.fullmatch(r"N\d", token):
        return token[1:]
    return token


def humanize_head(head: str) -> str:
    return head.upper()


def split_binding(binding: str) -> tuple[str, list[str]]:
    tokens = normalize_space(binding).split(" ")
    if not tokens:
        return "", []
    head = tokens[0].lstrip("&")
    return head, tokens[1:]


def binding_to_lines(binding: str) -> list[str]:
    binding = normalize_space(binding)
    if binding == "&trans":
        return ["TRNS"]
    if binding == "&none":
        return ["NONE"]

    head, args = split_binding(binding)
    if head == "kp" and args:
        return [normalize_keycode(args[0])]
    if head == "mkp" and args:
        return [normalize_keycode(args[0])]
    if head == "bt" and args:
        if args[0] == "BT_SEL" and len(args) > 1:
            return [f"BT {args[1]}"]
        if args[0] == "BT_CLR_ALL":
            return ["BT", "CLR ALL"]
        if args[0] == "BT_CLR":
            return ["BT CLR"]
        return ["BT", " ".join(args)]
    if head == "bootloader":
        return ["BOOT"]
    if head in {"mo", "to", "sl"} and args:
        return [f"{head.upper()} {args[0]}"]
    if head == "mt" and len(args) >= 2:
        return [normalize_keycode(args[-1]), normalize_keycode(args[0])]
    if head == "lt" and len(args) >= 2:
        return [normalize_keycode(args[-1]), f"LT {args[0]}"]
    if len(args) >= 2:
        return [normalize_keycode(args[-1]), f"{humanize_head(head)} {' '.join(args[:-1])}".strip()]
    if len(args) == 1:
        return [normalize_keycode(args[0]), humanize_head(head)]
    return [humanize_head(head)]


def binding_to_summary(binding: str) -> str:
    lines = binding_to_lines(binding)
    return " / ".join(line for line in lines if line)


def describe_macro_step(step: str) -> str:
    step = normalize_space(step)
    if step in MACRO_CONTROL_LABELS:
        return MACRO_CONTROL_LABELS[step]
    if step.startswith("&macro_param_"):
        return step
    return binding_to_summary(step)


def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    if not text:
        return [""]
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            wrapped_lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
                continue
            if current:
                wrapped_lines.append(current)
                current = character
            else:
                wrapped_lines.append(candidate)
                current = ""
        if current:
            wrapped_lines.append(current)
    return wrapped_lines


def draw_centered_text_lines(
    pdf: canvas.Canvas,
    lines: Iterable[str],
    center_x: float,
    center_y: float,
    max_width: float,
    primary_font_size: float = 11,
    secondary_font_size: float = 8,
) -> None:
    lines = [line for line in lines if line]
    if not lines:
        return

    spacing = 12 if len(lines) == 1 else 10
    start_y = center_y + ((len(lines) - 1) * spacing) / 2
    for index, line in enumerate(lines):
        font_size = primary_font_size if index == 0 else secondary_font_size
        while font_size > 5 and pdfmetrics.stringWidth(line, "Helvetica", font_size) > max_width:
            font_size -= 0.5
        pdf.setFont("Helvetica", font_size)
        text_width = pdfmetrics.stringWidth(line, "Helvetica", font_size)
        pdf.drawString(center_x - text_width / 2, start_y - index * spacing, line)


def draw_layer_page(
    pdf: canvas.Canvas,
    title: str,
    layer: Layer,
    layout_keys: list[LayoutKey],
) -> None:
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(MARGIN, PAGE_HEIGHT - MARGIN, title)

    pdf.setFillColor(colors.HexColor("#475569"))
    pdf.setFont("Helvetica", 10)
    pdf.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 18, f"Layer node: {layer.name}")

    max_x = max(key.x + key.width for key in layout_keys)
    max_y = max(key.y + key.height for key in layout_keys)
    available_width = PAGE_WIDTH - (MARGIN * 2)
    available_height = PAGE_HEIGHT - TITLE_GAP - MARGIN - BOTTOM_MARGIN
    cell_size = min(available_width / max_x, available_height / max_y)
    gap = max(cell_size * 0.08, 4)
    origin_x = (PAGE_WIDTH - (max_x * cell_size)) / 2
    top_y = PAGE_HEIGHT - TITLE_GAP - MARGIN

    for index, key in enumerate(layout_keys):
        binding = layer.bindings[index]
        box_x = origin_x + key.x * cell_size + gap / 2
        box_y = top_y - (key.y + key.height) * cell_size + gap / 2
        box_width = key.width * cell_size - gap
        box_height = key.height * cell_size - gap

        fill = colors.HexColor("#ffffff")
        stroke = colors.HexColor("#94a3b8")
        if binding == "&trans":
            fill = colors.HexColor("#f8fafc")
        elif binding == "&none":
            fill = colors.HexColor("#f1f5f9")

        pdf.setFillColor(fill)
        pdf.setStrokeColor(stroke)
        pdf.roundRect(box_x, box_y, box_width, box_height, 8, fill=1, stroke=1)
        pdf.setFillColor(colors.HexColor("#0f172a"))
        draw_centered_text_lines(
            pdf,
            binding_to_lines(binding),
            center_x=box_x + box_width / 2,
            center_y=box_y + box_height / 2 - 4,
            max_width=box_width - 10,
        )

    pdf.setFillColor(colors.HexColor("#64748b"))
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(PAGE_WIDTH - MARGIN, BOTTOM_MARGIN - 8, "Generated from config/Pyuron.keymap")
    pdf.showPage()


def lookup_description(entries: dict[str, str], *names: str | None) -> str | None:
    for name in names:
        if name and name in entries:
            return entries[name]
    return None


def default_combo_description(combo: Combo, base_layer: Layer) -> str:
    trigger_keys = [binding_to_summary(base_layer.bindings[position]) for position in combo.positions]
    return f"Trigger with {' + '.join(trigger_keys)} and send {binding_to_summary(combo.binding)}."


def default_macro_description(macro: Macro) -> str:
    steps = " -> ".join(describe_macro_step(step) for step in macro.steps)
    return f"Sequence: {steps}"


def estimate_entry_height(lines: list[str]) -> float:
    return 30 + (len(lines) * 13)


def draw_notes_page(
    pdf: canvas.Canvas,
    heading: str,
    entries: list[tuple[str, list[str]]],
) -> None:
    if not entries:
        return

    y = PAGE_HEIGHT - MARGIN

    def start_page() -> float:
        pdf.setFillColor(colors.HexColor("#0f172a"))
        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(MARGIN, PAGE_HEIGHT - MARGIN, heading)
        pdf.setFillColor(colors.HexColor("#475569"))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 18, "Generated automatically from the ZMK keymap sources")
        return PAGE_HEIGHT - MARGIN - 46

    y = start_page()

    for title, raw_lines in entries:
        wrapped_lines: list[str] = []
        for line in raw_lines:
            wrapped_lines.extend(wrap_text(line, "Helvetica", 10, PAGE_WIDTH - (MARGIN * 2)))
        needed_height = estimate_entry_height(wrapped_lines)
        if y - needed_height < BOTTOM_MARGIN:
            pdf.showPage()
            y = start_page()

        pdf.setFillColor(colors.HexColor("#e2e8f0"))
        pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
        pdf.roundRect(MARGIN, y - needed_height + 8, PAGE_WIDTH - (MARGIN * 2), needed_height, 10, fill=1, stroke=1)

        cursor_y = y - 18
        pdf.setFillColor(colors.HexColor("#0f172a"))
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(MARGIN + 14, cursor_y, title)

        cursor_y -= 20
        pdf.setFont("Helvetica", 10)
        for line in wrapped_lines:
            pdf.drawString(MARGIN + 14, cursor_y, line)
            cursor_y -= 13

        y -= needed_height + 12

    pdf.showPage()


def build_combo_entries(
    combos: list[Combo],
    combo_notes: dict[str, str],
    base_layer: Layer,
) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    for combo in combos:
        description = lookup_description(combo_notes, combo.name, combo.label) or default_combo_description(combo, base_layer)
        trigger_keys = [binding_to_summary(base_layer.bindings[position]) for position in combo.positions]
        lines = [
            f"Note: {description}",
            f"Trigger keys: {' + '.join(trigger_keys)}",
            f"Key positions: {', '.join(str(position) for position in combo.positions)}",
            f"Output binding: {normalize_space(combo.binding)}",
        ]
        entries.append((combo.label or combo.name, lines))
    return entries


def build_macro_entries(
    macros: list[Macro],
    macro_notes: dict[str, str],
    behaviors: list[Behavior],
) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    for macro in macros:
        description = lookup_description(macro_notes, macro.name, macro.label) or default_macro_description(macro)
        referenced_by = [
            behavior.label or behavior.name
            for behavior in behaviors
            if any(binding.startswith(f"&{macro.name}") for binding in behavior.bindings)
        ]
        lines = [f"Note: {description}"]
        if referenced_by:
            lines.append(f"Referenced by: {', '.join(referenced_by)}")
        lines.append("Sequence:")
        lines.extend(f"  - {normalize_space(step)}" for step in macro.steps)
        entries.append((macro.label or macro.name, lines))
    return entries


def load_metadata(metadata_path: Path | None) -> dict:
    if not metadata_path or not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def generate_pdf(
    output_path: Path,
    title: str,
    layout_keys: list[LayoutKey],
    layers: list[Layer],
    combos: list[Combo],
    macros: list[Macro],
    behaviors: list[Behavior],
    layer_names: dict[str, str],
    combo_notes: dict[str, str],
    macro_notes: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=PAGE_SIZE)
    base_layer = next((layer for layer in layers if layer.name == "default_layer"), layers[0])

    for layer in layers:
        layer_title = layer_names.get(layer.name, layer.name)
        draw_layer_page(pdf, f"{title} - {layer_title}", layer, layout_keys)

    draw_notes_page(pdf, "Combo notes", build_combo_entries(combos, combo_notes, base_layer))
    draw_notes_page(pdf, "Macro notes", build_macro_entries(macros, macro_notes, behaviors))
    pdf.save()


def validate_layout_and_layers(layout_keys: list[LayoutKey], layers: list[Layer]) -> None:
    key_count = len(layout_keys)
    for layer in layers:
        if len(layer.bindings) != key_count:
            raise ValueError(
                f"Layer '{layer.name}' has {len(layer.bindings)} bindings but the layout expects {key_count}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF visualization for a ZMK keymap")
    parser.add_argument("--keymap", type=Path, required=True, help="Path to the .keymap file")
    parser.add_argument("--layout", type=Path, required=True, help="Path to the layout JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output PDF")
    parser.add_argument("--metadata", type=Path, help="Optional JSON file with titles and note overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    title = metadata.get("title", "Keymap")

    keymap_text = strip_line_comments(args.keymap.read_text(encoding="utf-8"))
    layout_keys = load_layout(args.layout)
    layers = parse_layers(keymap_text)
    combos = parse_combos(keymap_text)
    macros = parse_macros(keymap_text)
    behaviors = parse_custom_behaviors(keymap_text)

    validate_layout_and_layers(layout_keys, layers)
    generate_pdf(
        output_path=args.output,
        title=title,
        layout_keys=layout_keys,
        layers=layers,
        combos=combos,
        macros=macros,
        behaviors=behaviors,
        layer_names=metadata.get("layer_names", {}),
        combo_notes=metadata.get("combos", {}),
        macro_notes=metadata.get("macros", {}),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
