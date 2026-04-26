import argparse
import sys
import os

def count_file_stats(filepath):
    """
    Reads a file and returns the number of lines and words.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    line_count = 0
    word_count = 0
    char_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
            num_lines = len(lines)
            num_words = len(" ".join(lines).split())
            char_count = len(" ".join(lines))
                
        
        return {
            "lines": num_lines,
            "words": num_words,
            "chars": char_count
        }
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Count lines and words in a text file.")
    parser.add_argument("file", help="Path to the text file to read.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed stats.")

    args = parser.parse_args()

    stats = count_file_stats(args.file)
    
    if stats:
        print(f"--- Statistics for: {args.file} ---")
        print(f"Lines: {stats['lines']}")
        print(f"Words: {stats['words']}")
        if args.verbose:
            print(f"Characters: {stats['chars']}")

if __name__ == "__main__":
    main()
