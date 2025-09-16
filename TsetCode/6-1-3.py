import os
import random

# Ratio for splitting the dataset: Train:Dev:Test = 6:1:3
TRAIN_RATIO = 6
DEV_RATIO = 1
TEST_RATIO = 3

# Path of the AiVoice metadata file
ai_metadata_path = r"U:\Code\HKBU\1\Code\aasist\aasist-main\HKBUxAIVoice_250713\metadata.txt"

# Output file names
output_train_path = r"CD.cn.train.trl.txt"
output_dev_path   = r"CD.cn.dev.trl.txt"
output_test_path  = r"CD.cn.test.trl.txt"

# A-codes range from A01 to A12 (only choose among A01-A12 randomly)
A_codes = [f"A{i:02d}" for i in range(1, 13)]

def transform_filename(original_filename, speaker, gender):
    """
    Transforms the original file name to new format:
      - Convert speaker to 2-digit string (e.g., 1 -> "01")
      - Gender is represented as its first letter in uppercase (e.g., "female" -> "F")
      - If original filename is of the form "hifigan_16k_001.wav",
        then it will be converted to "hifigan_01_F_001.wav".
      - If original filename is of the form "ar_001.wav",
        then it will be converted to "ar_01_F_001.wav".
    """
    # Format speaker and gender code
    try:
        speaker_str = f"{int(speaker):02d}"
    except ValueError:
        speaker_str = speaker
    gender_code = gender.strip()[0].upper()
    
    parts = original_filename.split("_")
    # If file name has three or more parts, assume pattern: prefix, (middle), rest
    if len(parts) >= 3:
        # Use the first part and the last part; ignore the original middle part(s)
        new_filename = f"{parts[0]}_{speaker_str}_{gender_code}_{parts[-1]}"
    elif len(parts) == 2:
        new_filename = f"{parts[0]}_{speaker_str}_{gender_code}_{parts[1]}"
    else:
        # Fallback: insert speaker and gender before the extension
        name, ext = os.path.splitext(original_filename)
        new_filename = f"{name}_{speaker_str}_{gender_code}{ext}"
    
    return new_filename

def parse_metadata(metadata_path):
    """
    Reads the metadata file and returns a list of tuples.
    Each tuple contains (base_id, new_file_no_ext, random_A_code, "spoof")
    where:
      - base_id: e.g. "hifigan_01" (derived from new file name by joining first two parts)
      - new_file_no_ext: new file name without the '.wav' extension
      - random_A_code: a randomly chosen code from A_codes list
      - "spoof": a constant string
    """
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            # Clean and skip empty lines
            line = line.strip()
            if not line:
                continue
            # Split each line by comma; expected format:
            # filename,content,speaker,gender
            parts = line.split(",")
            if len(parts) < 4:
                print(f"Line does not have enough fields: {line}")
                continue
                
            original_filename, content, speaker, gender = parts[0], parts[1], parts[2], parts[3]
            # Transform the filename using speaker id and gender
            new_filename = transform_filename(original_filename, speaker, gender)
            # Remove extension for second column
            new_filename_no_ext, _ = os.path.splitext(new_filename)
            # Determine base_id from new_filename_no_ext:
            # For example, "hifigan_01_F_001" -> base_id = "hifigan_01"
            new_parts = new_filename_no_ext.split("_")
            if len(new_parts) >= 2:
                base_id = f"{new_parts[0]}_{new_parts[1]}"
            else:
                base_id = new_filename_no_ext
                
            # Randomly choose an A-code from A_codes (A01 ~ A12)
            random_A_code = random.choice(A_codes)
            
            entries.append((base_id, new_filename_no_ext, random_A_code, "spoof"))
    return entries

def split_entries(entries, train_ratio, dev_ratio, test_ratio):
    """
    Randomly shuffles and splits the entries list into train, dev, and test sets
    according to the provided ratios.
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    total = len(entries)
    
    # Shuffle the entries so that the split is random
    random.shuffle(entries)
    
    train_count = int(total * train_ratio / total_ratio)
    dev_count = int(total * dev_ratio / total_ratio)
    test_count = total - train_count - dev_count  # assign remaining to test
    
    train_set = entries[0:train_count]
    dev_set = entries[train_count:train_count+dev_count]
    test_set = entries[train_count+dev_count:]
    
    return train_set, dev_set, test_set

def write_entries(entries, output_path):
    """
    Writes the entries to an output file.
    Each line is in the format:
      base_id new_file_no_ext A_code spoof
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            line = " ".join(entry)
            f.write(line + "\n")
    print(f"Results written to: {output_path}")

def main():
    # Parse metadata file and generate the list of entries with transformed file names
    entries = parse_metadata(ai_metadata_path)
    
    if not entries:
        print("No metadata entries found. Exiting.")
        return
    
    # Split entries into training, dev, and test sets
    train_set, dev_set, test_set = split_entries(entries, TRAIN_RATIO, DEV_RATIO, TEST_RATIO)
    
    # Write the results to corresponding output files
    write_entries(train_set, output_train_path)
    write_entries(dev_set, output_dev_path)
    write_entries(test_set, output_test_path)

if __name__ == "__main__":
    main()