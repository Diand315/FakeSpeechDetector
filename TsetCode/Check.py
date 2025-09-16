import os
import wave
import contextlib
from tqdm import tqdm 

# Set the path to the directory containing the processed New CD audio files.
new_cd_folder_path = r"U:\Code\HKBU\1\Code\aasist\aasist-main\CantoneseVoice\Real_People"

# Set the path for the output TXT file.
output_txt_path = r"U:\Code\HKBU\1\Code\aasist\aasist-main\CantoneseVoice\CD_audio_analysis.txt"

# Dictionary to hold the durations per group.
# For each key (group) the value is a list of durations.
group_durations = {}

# Dictionary to hold the count of files with duration > 4 seconds per group.
group_long_counts = {}

# List to store files that have a duration longer than 4 seconds (overall list).
long_files = []

# Get all .wav files in the directory for processing.
wav_files = [filename for filename in os.listdir(new_cd_folder_path) if filename.lower().endswith(".wav")]

# Iterate over each file with a progress bar.
for filename in tqdm(wav_files, desc="Processing WAV files"):
    file_path = os.path.join(new_cd_folder_path, filename)
    try:
        # Open the WAV file and calculate its duration.
        with contextlib.closing(wave.open(file_path, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
    except wave.Error as e:
        print(f"Error reading {filename}: {e}")
        continue
    except Exception as e:
        print(f"Unable to process {filename}: {e}")
        continue

    # Generate the grouping key from the filename.
    # For example, "CD1_01_M_99.wav" will be grouped as "CD1_01_M".
    parts = filename.split("_")
    if len(parts) < 2:
        print(f"Filename format not recognized for grouping: {filename}")
        continue
    group_key = "_".join(parts[:3])
    group_durations.setdefault(group_key, []).append(duration)

    # 更新每個群組中超過 4 秒的檔案計數
    if duration > 4.0:
        group_long_counts[group_key] = group_long_counts.get(group_key, 0) + 1
        long_files.append((filename, duration))
    else:
        # 若該群組未記錄過，則設為0（以便後續計算百分比時不失效）
        if group_key not in group_long_counts:
            group_long_counts[group_key] = 0

# Prepare the output text content.
output_lines = []
output_lines.append("Average durations for each group:")

for group_key in sorted(group_durations.keys()):
    durations = group_durations[group_key]
    total_count = len(durations)
    avg_duration = sum(durations) / total_count
    # 計算群組中超過4秒檔案所佔百分比
    long_count = group_long_counts.get(group_key, 0)
    long_percentage = (long_count / total_count * 100) if total_count else 0
    output_lines.append(
        f"{group_key}: average duration = {avg_duration:.2f} seconds ({total_count} files) {long_percentage:.0f}%"
    )

output_lines.append("\nFiles longer than 4 seconds (overall):")
if long_files:
    for fname, dur in long_files:
        output_lines.append(f"{fname} - {dur:.2f} seconds")
else:
    output_lines.append("No files longer than 4 seconds.")

# Write the results to the output TXT file.
with open(output_txt_path, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"Analysis results have been written to: {output_txt_path}")