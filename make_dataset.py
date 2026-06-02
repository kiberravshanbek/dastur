"""
A0/A1 uchun YOLO datasetini OpenCV annotatsiyasi orqali avtomatik yaratish.
Ikki dataset: area (bloklar) va circle (doirachalar).
"""
import sys, os, glob, random
sys.path.insert(0, '/home/claude/pmtests_pro')
import omr_opencv as P
import cv2, numpy as np
from pdf2image import convert_from_path
from pathlib import Path

SRC = "/tmp/namuna/namuna"
OUT = "/home/claude/training/data"
DPI = 150

# Papka strukturasi
for ds in ["area", "circle"]:
    for split in ["train", "val"]:
        os.makedirs(f"{OUT}/{ds}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT}/{ds}/labels/{split}", exist_ok=True)


def yolo_box(x1, y1, x2, y2, W, H):
    """Piksel bbox -> YOLO normalize (cx, cy, w, h)."""
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = abs(x2 - x1) / W
    h = abs(y2 - y1) / H
    return cx, cy, w, h


def process_page(img):
    """Sahifadan area (bloklar) va circle (doirachalar) annotatsiyalarini olish.
    Returns: (area_boxes, circle_boxes) yoki None agar ishonchsiz.
    area_boxes: [(x1,y1,x2,y2), ...] 4 ta blok
    circle_boxes: [(x1,y1,x2,y2), ...] barcha doirachalar
    """
    table = P._find_answer_table(img)
    x, y, w, h = table
    bubbles, bw = P._detect_bubbles(img, x, y, x+w, y+h)
    if len(bubbles) < 200:
        return None
    try:
        cols = P._find_columns(bubbles, n_blocks=4, opts_per_block=4)
    except Exception:
        return None
    if len(cols) != 16:
        return None

    # 4 blok chegaralari (har 4 ustun = 1 blok)
    # Blok kengligi ustun oraliqlaridan, balandligi barcha bubble Y dan
    all_y1 = min(b[1]-b[2] for b in bubbles) - 10
    all_y2 = max(b[1]+b[2] for b in bubbles) + 10
    # O'rtacha ustun oralig'i
    col_step = np.median([cols[i+1]-cols[i] for i in range(15)
                          if cols[i+1]-cols[i] < 200])
    area_boxes = []
    for bi in range(4):
        crange = cols[bi*4:bi*4+4]
        bx1 = crange[0] - col_step*0.6
        bx2 = crange[-1] + col_step*0.6
        # Har blokning o'z bubble'lari bo'yicha Y chegarasi (aniqroq)
        cmin = crange[0] - 30
        cmax = crange[-1] + 30
        blk_bubbles = [b for b in bubbles if cmin <= b[0] <= cmax]
        if len(blk_bubbles) < 10:
            return None
        by1 = min(b[1]-b[2] for b in blk_bubbles) - 8
        by2 = max(b[1]+b[2] for b in blk_bubbles) + 8
        area_boxes.append((bx1, by1, bx2, by2))

    # Doirachalar (bubble = (cx, cy, r))
    circle_boxes = [(b[0]-b[2], b[1]-b[2], b[0]+b[2], b[1]+b[2]) for b in bubbles]

    return area_boxes, circle_boxes


def main():
    files = sorted(glob.glob(f"{SRC}/*.pdf"))
    samples = []  # (image, area_boxes, circle_boxes)
    print(f"Fayllar: {len(files)}")

    for f in files:
        try:
            pages = convert_from_path(f, dpi=DPI)
        except Exception as e:
            print(f"  PDF xato {f}: {e}")
            continue
        for pi, page in enumerate(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            res = process_page(img)
            if res is None:
                continue
            area_boxes, circle_boxes = res
            samples.append((img, area_boxes, circle_boxes))

    print(f"Ishonchli sahifalar: {len(samples)}")

    # Train/val split (85/15)
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.15))
    val_set = samples[:n_val]
    train_set = samples[n_val:]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    for split, dataset in [("train", train_set), ("val", val_set)]:
        for idx, (img, area_boxes, circle_boxes) in enumerate(dataset):
            H, W = img.shape[:2]
            name = f"page_{split}_{idx:04d}"

            # AREA dataset
            cv2.imwrite(f"{OUT}/area/images/{split}/{name}.jpg", img,
                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(f"{OUT}/area/labels/{split}/{name}.txt", "w") as fp:
                for (x1,y1,x2,y2) in area_boxes:
                    cx,cy,bw,bh = yolo_box(x1,y1,x2,y2,W,H)
                    fp.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # CIRCLE dataset
            cv2.imwrite(f"{OUT}/circle/images/{split}/{name}.jpg", img,
                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(f"{OUT}/circle/labels/{split}/{name}.txt", "w") as fp:
                for (x1,y1,x2,y2) in circle_boxes:
                    cx,cy,bw,bh = yolo_box(x1,y1,x2,y2,W,H)
                    fp.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # YAML config fayllar
    for ds, cls_name in [("area", "block"), ("circle", "bubble")]:
        with open(f"{OUT}/{ds}/data.yaml", "w") as fp:
            fp.write(f"path: {OUT}/{ds}\n")
            fp.write("train: images/train\n")
            fp.write("val: images/val\n")
            fp.write("nc: 1\n")
            fp.write(f"names: ['{cls_name}']\n")

    print("Dataset tayyor!")
    print(f"  area:   {OUT}/area/data.yaml")
    print(f"  circle: {OUT}/circle/data.yaml")


if __name__ == "__main__":
    main()
