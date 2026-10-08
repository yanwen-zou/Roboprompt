#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-json", required=True, help="Path to meta/info.json")
    parser.add_argument("--old-pattern", required=True, help="Expected current video_path pattern")
    parser.add_argument("--new-pattern", required=True, help="Replacement video_path pattern")
    args = parser.parse_args()

    info_json_path = Path(args.info_json)
    with open(info_json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    current_video_path = info.get("video_path")
    if current_video_path == args.new_pattern:
        print(f"video_path already updated: {info_json_path}")
        return

    if current_video_path != args.old_pattern:
        print(
            f"Skipping video_path update for {info_json_path}: "
            f"unexpected current value {current_video_path!r}"
        )
        return

    info["video_path"] = args.new_pattern
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")
    print(f"Updated video_path in {info_json_path}")


if __name__ == "__main__":
    main()
