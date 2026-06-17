from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    ratio = width / max(1, image.width)
    height = max(1, int(image.height * ratio))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def find_case_dir(case_root: Path, index: int, case: Dict[str, Any]) -> Path:
    matches = sorted(case_root.glob(f"{index:02d}_{case['case_group']}_{case['id']}"))
    if not matches:
        raise FileNotFoundError(f"Case image directory not found for case {index}: {case_root}")
    return matches[0]


def find_demo_dir(demo_root: Path, index: int, case: Dict[str, Any]) -> Path:
    matches = sorted(demo_root.glob(f"{index:02d}_{case['case_group']}_{case['id']}"))
    if not matches:
        raise FileNotFoundError(
            "Result image directory not found for "
            f"case {index}. Run scripts/record_qwen_case_demo.sh --all first."
        )
    return matches[0]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + max(0, (right - left - (bbox[2] - bbox[0])) // 2)
    y = top + max(0, (bottom - top - (bbox[3] - bbox[1])) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def render_grid(
    cases: Sequence[Dict[str, Any]],
    case_root: Path,
    demo_root: Path,
    output_path: Path,
    result_source: str,
) -> None:
    rows = list(cases)
    if not rows:
        raise ValueError("No cases selected for rendering.")

    col_w = 300
    img_h = 230
    row_title_h = 34
    col_title_h = 32
    gap_x = 16
    gap_y = 18
    margin_x = 28
    margin_y = 20
    header_font = load_font(20)
    object_font = load_font(18)

    width = margin_x * 2 + col_w * 3 + gap_x * 2
    height = margin_y * 2 + col_title_h + len(rows) * (row_title_h + img_h) + (len(rows) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["RGB", "Depth", "Result"]
    for col, header in enumerate(headers):
        x = margin_x + col * (col_w + gap_x)
        draw_centered(draw, header, (x, margin_y, x + col_w, margin_y + col_title_h), header_font, (35, 35, 35))

    result_name = "after_oracle_rgb.png" if result_source == "oracle" else "after_prediction_rgb.png"
    y = margin_y + col_title_h
    for idx, case in enumerate(rows, start=1):
        case_dir = find_case_dir(case_root, idx, case)
        demo_dir = find_demo_dir(demo_root, idx, case)
        image_paths = [
            case_dir / "initial_rgb.png",
            case_dir / "initial_depth.png",
            demo_dir / result_name,
        ]
        missing = [str(path) for path in image_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing image(s): " + ", ".join(missing))

        object_name = str(case.get("object_type") or "Object")
        draw_centered(
            draw,
            object_name,
            (margin_x, y, width - margin_x, y + row_title_h),
            object_font,
            (25, 25, 25),
        )
        row_img_y = y + row_title_h
        for col, path in enumerate(image_paths):
            image = fit_image(Image.open(path), (col_w, img_h))
            x = margin_x + col * (col_w + gap_x)
            canvas.paste(image, (x, row_img_y))
        y += row_title_h + img_h + gap_y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=Path("runs/case_studies_10k_stay_t200"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/case_studies_10k_stay_t200/panels"))
    parser.add_argument("--demo-dir", type=Path, default=Path("runs/qwen_case_demos"))
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--output-name", default="case_study_grid.png")
    parser.add_argument("--result-source", choices=["prediction", "oracle"], default="prediction")
    args = parser.parse_args()

    cases = json.loads((args.case_dir / "cases.json").read_text(encoding="utf-8"))
    cases = cases[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / args.output_name
    render_grid(
        cases,
        case_root=args.case_dir,
        demo_root=args.demo_dir,
        output_path=output_path,
        result_source=args.result_source,
    )
    (args.output_dir / "panels.md").write_text(f"# Case Study Grid\n\n![Case Study Grid]({args.output_name})\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "case_count": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
