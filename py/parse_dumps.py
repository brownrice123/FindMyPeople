import zstandard as zstd
import json
import csv
import os
import sys

DUMPS_DIR = os.path.join(os.path.dirname(__file__), "..", "torrents", "subreddits25")
CSV_DIR   = os.path.join(os.path.dirname(__file__), "..", "csv")


def parse_dump(input_path, output_path):
    """
    Stream-parse a Pushshift .zst comment dump.
    Extract (author, subreddit) pairs and write to CSV.
    """
    row_count = 0
    skip_count = 0

    with open(input_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        with dctx.stream_reader(fh) as reader:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["author", "subreddit"])  # header

                buffer = b""
                while True:
                    chunk = reader.read(2**24)  # 16MB chunks
                    if not chunk:
                        break

                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines[-1]  # keep incomplete last line

                    for line in lines[:-1]:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            author = obj.get("author", "")
                            subreddit = obj.get("subreddit", "")

                            # skip deleted/bot accounts
                            if author in ("", "[deleted]", "AutoModerator"):
                                skip_count += 1
                                continue

                            writer.writerow([author, subreddit])
                            row_count += 1

                            if row_count % 100_000 == 0:
                                print(f"  {row_count:,} rows written...")

                        except json.JSONDecodeError:
                            skip_count += 1
                            continue

    print(f"Done: {row_count:,} rows written, {skip_count:,} skipped")
    print(f"Output: {output_path}")


def process_file(zst_path):
    stem = os.path.basename(zst_path).removesuffix(".zst")
    csv_path = os.path.join(CSV_DIR, f"{stem}.csv")

    print(f"\nParsing {os.path.basename(zst_path)}...")
    parse_dump(zst_path, csv_path)
    os.remove(zst_path)
    print(f"Deleted {os.path.basename(zst_path)}")


if __name__ == "__main__":
    os.makedirs(CSV_DIR, exist_ok=True)

    if len(sys.argv) == 2:
        fname = sys.argv[1]
        if not fname.endswith(".zst"):
            fname += ".zst"
        zst_path = os.path.join(DUMPS_DIR, fname)
        if not os.path.exists(zst_path):
            print(f"Error: file not found: {zst_path}")
            sys.exit(1)
        process_file(zst_path)
    elif len(sys.argv) == 1:
        zst_files = sorted(
            os.path.join(DUMPS_DIR, f)
            for f in os.listdir(DUMPS_DIR)
            if f.endswith(".zst")
        )
        if not zst_files:
            print(f"No .zst files found in {DUMPS_DIR}")
            sys.exit(0)
        print(f"Found {len(zst_files)} .zst file(s) to process.")
        for zst_path in zst_files:
            process_file(zst_path)
        print("\nAll files processed.")
    else:
        print("Usage: python parse_dumps.py [filename.zst]")
        print("  No argument  — process all .zst files in torrents/subreddits25/")
        print("  filename.zst — process that one file (name only, no path needed)")
        sys.exit(1)