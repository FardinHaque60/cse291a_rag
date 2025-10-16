# This script converts a CSV file to a JSON file.
# Usage: python convert_csv_to_json.py <input_csv_path> <output_json_path>
import csv
import json
import sys

def convert_csv_to_json(csv_path, json_path):
    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)
    with open(json_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_to_json.py <input_csv_path> <output_json_path>")
        sys.exit(1)
    convert_csv_to_json(sys.argv[1], sys.argv[2])