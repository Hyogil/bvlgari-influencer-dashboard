from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--instagram", default="avatars_96", help="Folder containing Instagram JPG avatars")
parser.add_argument("--tiktok", default="avatars_96_tiktok", help="Folder containing TikTok JPG avatars")
parser.add_argument("--youtube", default="", help="Optional folder containing YouTube JPG avatars")
parser.add_argument("--project", default=".", help="Project root containing Resource/")
args = parser.parse_args()

project = Path(args.project).resolve()

targets = [
    ("instagram", Path(args.instagram) if args.instagram else None),
    ("tiktok", Path(args.tiktok) if args.tiktok else None),
    ("youtube", Path(args.youtube) if args.youtube else None),
]

for platform, source in targets:
    if source is None:
        continue
    source = source.resolve()
    dest = project / "Resource" / "images" / platform
    dest.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"[{platform}] source folder not found: {source}")
        continue

    count = 0
    for p in source.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            # Dashboard expects .jpg; Selenium scripts already output JPG.
            if p.suffix.lower() == ".jpg":
                out = dest / p.name
                shutil.copy2(p, out)
                count += 1
            else:
                print(f"[{platform}] skipped non-JPG file: {p.name}")
    print(f"[{platform}] copied {count} JPG files -> {dest}")

print("Done.")
