import zstandard as zstd
import json
import csv
import os
import sys

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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_dumps.py <input.zst> <output.csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)
    
    print(f"Parsing {input_path}...")
    parse_dump(input_path, output_path)