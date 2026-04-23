import pandas as pd
import sys
import os

def convert_parquet_to_excel(input_path):
    # Output file path
    output_path = os.path.splitext(input_path)[0] + ".xlsx"
    
    print(f"Reading: {input_path}")
    try:
        # parquet load
        df = pd.read_parquet(input_path)
        
        # Excel saving
        df.to_excel(output_path, index=False)
        
        print(f"Conversion completed! File saved in: {output_path}")
        print(f"Total rows: {len(df)}")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_parquet.py <parquet_file_path>")
    else:
        convert_parquet_to_excel(sys.argv[1])