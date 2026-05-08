import json
from pathlib import Path
from PIL import Image, ImageDraw

IMAGE_DIR = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd/images")
JSON_DIR = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd/jsons")
OUTPUT_DIR = Path("results") / "anotated"

TARGET_IDS = ["3114"]


def annotate_specific_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_id in TARGET_IDS:
        img_path = IMAGE_DIR / f"{image_id}.jpg"
        json_path = JSON_DIR / f"{image_id}.json"

        # Check if both files exist before processing
        if not img_path.exists():
            print(f"Skipping {image_id}: Image not found at {img_path}")
            continue

        if not json_path.exists():
            print(f"Skipping {image_id}: JSON not found at {json_path}")
            continue

        # 1. Load the Ground Truth points
        with open(json_path, 'r') as f:
            data = json.load(f)
            points = data.get("points", [])
            human_num = data.get("human_num", 0)

        # 2. Open image and prepare to draw
        # Use a 'with' block to ensure the file is closed properly
        with Image.open(img_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)

            # 3. Draw dots for each person
            # Radius of 4 creates a visible 9x9 pixel dot
            radius = 4
            for x, y in points:
                # Bounding box for the circle: [x0, y0, x1, y1]
                shape = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(shape, fill="red", outline="white", width=1)

            # 4. Save the result
            save_path = OUTPUT_DIR / f"{image_id}_gt.jpg"
            img.save(save_path, quality=95)  # High quality JPEG
            print(f"Successfully annotated {image_id} (Count: {human_num})")


if __name__ == "__main__":
    annotate_specific_images()