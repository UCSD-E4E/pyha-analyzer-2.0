import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import csv
import soundfile as sf
from scipy import signal
from datetime import datetime

# Directories for template audio, clips to make detections in, and output results

CLIP_PATH = "/home/super/data/music/Location A (Sand Forrest)/A Zoom F3_03-05-25"
TEMPLATE_PATH = "/home/super/data/music/templates"
OUTPUT_DIR = "/home/super/template_matching_results"

# Customizable threshold for detections
THRESHOLD = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output file paths for results and CSV file

# The CSV file will contain all matches found during the template matching process, showing the name of the template, 
# the name of the clip, the timestamp of the match, and the score of the match.

OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"results_{timestamp}.txt")
CSV_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"all_matches_{timestamp}.csv")

# Computes mel spectrogram from an audio signal

def compute_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Converts a mel spectrogram in dB to an image format suitable for use in template matching or visualization

def spectrogram_to_image(S_dB):
    img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    img = (img * 255).astype(np.uint8)
    return img

# Finds the dominant frequency range in a mel spectrogram based on the energy distribution
# Returns the mel bin indices that contain most of the energy

def find_dominant_frequency_range(spectrogram, energy_threshold=0.1):
    freq_energy = np.mean(spectrogram, axis=1)
    freq_energy = (freq_energy - freq_energy.min()) / (freq_energy.max() - freq_energy.min())
    
    dominant_bins = np.where(freq_energy > energy_threshold)[0]
    
    if len(dominant_bins) == 0:
        return 0, spectrogram.shape[0]

    return dominant_bins.min(), dominant_bins.max() + 1

# Extract a specific frequency range from the spectrogram to focus matching on relevant frequencies

def filter_spectrogram_by_frequency_range(spectrogram, freq_min, freq_max):
    return spectrogram[freq_min:freq_max, :]

# Efficiently load audio files with decimation to improve time

def fast_audio_load(audio_path, target_sr=22050):
    y, original_sr = sf.read(audio_path)
    
    if y.ndim > 1:
        y = y[:, 0]
    
    if original_sr != target_sr:
        decimation = original_sr // target_sr
        if decimation > 1 and len(y) > 100:
            try:
                y = signal.decimate(y, decimation)
                sr = original_sr // decimation
            except ValueError:
                y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
                sr = target_sr
        else:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
            sr = target_sr
    else:
        sr = target_sr
    
    return y, sr

# Get species name from xeno-canto filename format

def get_species_name(filename):
    base_name = filename.replace('.mp3', '').replace('.MP3', '').replace('.wav', '').replace('.WAV', '')
    parts = base_name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base_name

# Initialize CSV file for storing all detection results
with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['template_name', 'clip_name', 'timestamp', 'score'])

total_matches_count = 0

# Use just one template per species to avoid redundancy
template_files = [file for file in os.listdir(TEMPLATE_PATH) if file.lower().endswith(('.mp3', '.wav'))]
species_templates = {}

for template_file in template_files:
    species = get_species_name(template_file)
    if species not in species_templates:
        species_templates[species] = template_file

selected_templates = list(species_templates.values())

print(f"{len(template_files)} total templates")
print(f"Filtered down to {len(selected_templates)} unique species templates")
for species, template in species_templates.items():
    print(f"  {species}: {template}")

# Write process information to output file
with open(OUTPUT_FILE, 'w') as f:
    f.write(f"Template Matching Results: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Clips directory: {CLIP_PATH}\n")
    f.write(f"Templates directory: {TEMPLATE_PATH}\n")
    f.write(f"Total templates: {len(template_files)}\n")
    f.write(f"Unique species templates: {len(selected_templates)}\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"Suppression distance: {SUPPRESSION_DISTANCE}\n")
    f.write(f"CSV output file: {CSV_OUTPUT_FILE}\n\n")
    
    f.write("Species template mapping:\n")
    for species, template in species_templates.items():
        f.write(f"  {species}: {template}\n")
    f.write("\n")
    
    # Iterate through each selected template
    template_count = 0
    for template_file in selected_templates:
            f.write(f"\n{'='*60}\n")
            f.write(f"PROCESSING TEMPLATE: {template_file}\n")
            f.write(f"{'='*60}\n\n")
            
            # Load and process the template audio with librosa
            template_path = os.path.join(TEMPLATE_PATH, template_file)
            try:
                y_template, sr_template = librosa.load(template_path, sr=None)
                template_spec = compute_mel_spectrogram(y_template, sr_template)
                
                # Find the frequency range where the template has the most energy
                freq_min, freq_max = find_dominant_frequency_range(template_spec)
                f.write(f"Template dominant frequency range: mel bins {freq_min}-{freq_max} (out of {template_spec.shape[0]})\n")
                # Filter spectrogram to dominant frequencies and convert to image for template matching
                template_spec_filtered = filter_spectrogram_by_frequency_range(template_spec, freq_min, freq_max)
                template_img = spectrogram_to_image(template_spec_filtered)
                
                f.write(f"Successfully loaded template: {template_file}\n")
                f.write(f"Template shape, full: {template_spec.shape}, filtered: {template_spec_filtered.shape}\n")
            except Exception as exception:
                f.write(f"ERROR: Failed to load template {template_file}: {str(exception)}\n")
                f.write(f"Skipping this template.\n\n")
                continue
            
            # Create output folder for the template's results
            template_name_clean = template_file.replace('.mp3', '').replace('.wav', '').replace('.MP3', '').replace('.WAV', '')
            template_folder = os.path.join(OUTPUT_DIR, template_name_clean)
            os.makedirs(template_folder, exist_ok=True)

            # Create visualization of the template spectrogram
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            im1 = ax1.imshow(template_spec, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'Full Template Spectrogram: {template_file}')
            ax1.set_xlabel('Time Frames')
            ax1.set_ylabel('Mel Frequency Bins')
            
            ax1.axhline(y=freq_min, color='red', linestyle='--', alpha=0.7, label=f'Freq range: {freq_min}-{freq_max}')
            ax1.axhline(y=freq_max-1, color='red', linestyle='--', alpha=0.7)
            ax1.legend()
            plt.colorbar(im1, ax=ax1, label='Power (dB)')
            
            im2 = ax2.imshow(template_spec_filtered, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_title(f'Filtered Template Spectrogram (Used for Matching): {template_file}')
            ax2.set_xlabel('Time Frames')
            ax2.set_ylabel(f'Mel Frequency Bins ({freq_min}-{freq_max})')
            plt.colorbar(im2, ax=ax2, label='Power (dB)')
            
            plt.tight_layout()
            template_spec_file = os.path.join(template_folder, f"template_spectrogram.png")
            plt.savefig(template_spec_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            f.write(f"Template spectrogram saved to: {template_spec_file}\n")
            
            # Track results for summary statistics
            clip_names = []
            match_counts = []
            
            # Process each audio clip in the clips directory
            for clip in os.listdir(CLIP_PATH):
                if clip.lower().endswith('.wav'):
                    f.write(f"Processing {clip}\n")
                    clip_path = os.path.join(CLIP_PATH, clip)
                    
                    try:
                        y_clip, sr_clip = fast_audio_load(clip_path, target_sr=22050)
                        f.write(f"Loaded clip: {clip} (duration: {len(y_clip)/sr_clip:.1f}s)\n")
                    except Exception as exception:
                        f.write(f"ERROR: Failed to load {clip}: {str(exception)}\n")
                        f.write(f"Skipping this clip.\n\n")
                        continue
                    
                    # Process clip using same frequency filtering as template
                    clip_spec = compute_mel_spectrogram(y_clip, sr_clip)
                    f.write(f"Clip shape (full): {clip_spec.shape}\n")
                    
                    clip_spec_filtered = filter_spectrogram_by_frequency_range(clip_spec, freq_min, freq_max)
                    clip_img = spectrogram_to_image(clip_spec_filtered)
                    f.write(f"Clip shape (filtered): {clip_spec_filtered.shape}\n")

                    # Perform template matching using OpenCV
                    res = cv.matchTemplate(clip_img, template_img, cv.TM_CCOEFF_NORMED)
                    f.write(f"Performed frequency-filtered template matching for {clip}\n")
                    
                    # Filter all matches above threshold
                    locations = np.where(res >= THRESHOLD)
                    matches = []
                    for pt in zip(*locations[::-1]):
                        score = res[pt[1], pt[0]]
                        matches.append((pt[1], pt[0], score))

                    matches.sort(key=lambda x: x[2], reverse=True)
                    f.write(f"Found {len(matches)} matches above threshold {THRESHOLD}\n")

                    # Save all matches to CSV with timestamps
                    seconds_per_col = 512 / sr_clip
                    with open(CSV_OUTPUT_FILE, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        for y, x, score in matches:
                            timestamp_match = x * seconds_per_col
                            csv_writer.writerow([template_name_clean, clip, timestamp_match, score])
                    
                    total_matches_count += len(matches)

                    # Set the suppression_distance to half of the length of the template
                    template_length_frames = template_img.shape[1]
                    suppression_distance = int(template_length_frames)
                    f.write(f"Suppression distance (frames): {suppression_distance}\n")

                    # Apply non-maximum suppression to avoid overlapping detections
                    selected = []
                    for y, x, score in matches:
                        if all(abs(x - xc) > suppression_distance for _, xc, _ in selected):
                            selected.append((y, x, score))

                    f.write(f"Kept {len(selected)} non-overlapping matches\n")
                    
                    clip_names.append(clip)
                    match_counts.append(len(selected))

                    # Create visualization showing matches on the spectrogram
                    clip_name_clean = clip.replace('.wav', '').replace('.WAV', '')
                    clip_folder = os.path.join(template_folder, clip_name_clean)
                    os.makedirs(clip_folder, exist_ok=True)

                    plt.figure(figsize=(12, 4))
                    vis = cv.cvtColor(clip_img, cv.COLOR_GRAY2BGR)

                    # Draw green rectangles around each detected match
                    for y, x, score in selected:
                        top_left = (x, y)
                        bottom_right = (x + template_img.shape[1], y + template_img.shape[0])
                        cv.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)

                        timestamp = x * seconds_per_col
                        f.write(f"  Match at {timestamp:.2f} sec (Score: {score:.3f})\n")

                    plt.imshow(vis, aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Frequency-Filtered Matches: {clip} vs {template_file} (Mel bins {freq_min}-{freq_max})")
                    clip_plot_file = os.path.join(clip_folder, f"main_matches.png")
                    plt.savefig(clip_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    f.write(f"Main visualization saved to: {clip_plot_file}\n")

                    # Create plots of the top 10 matches
                    top_matches = selected[:10]
                    margin = 20

                    if top_matches:
                        f.write("Top 10 Match Details:\n")
                        for idx, (y, x, score) in enumerate(top_matches, 1):
                            # Get a zoomed region around each match
                            x1 = max(0, x - margin)
                            x2 = min(clip_img.shape[1], x + template_img.shape[1] + margin)
                            y1 = max(0, y - margin)
                            y2 = min(clip_img.shape[0], y + template_img.shape[0] + margin)

                            zoom_region = clip_img[y1:y2, x1:x2]

                            plt.figure(figsize=(6, 4))
                            plt.imshow(zoom_region, aspect='auto', origin='lower', cmap='viridis')
                            timestamp_match = x * seconds_per_col
                            plt.title(f"Match {idx} at {timestamp_match:.2f}s (Score: {score:.3f})")
                            plt.axis('off')
                            
                            match_file = os.path.join(clip_folder, f"match_{idx:02d}.png")
                            plt.savefig(match_file, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            f.write(f"  Match {idx}: {timestamp_match:.2f}s (Score: {score:.3f}) - Saved to: {match_file}\n")
                    
                    f.write(f"All files for {clip} saved to folder: {clip_folder}\n")
                    f.write("-" * 40 + "\n\n")

            # Create summary bar chart showing match counts across all clips
            if clip_names:
                plt.figure(figsize=(max(12, len(clip_names) * 0.8), 6))
                bars = plt.bar(range(len(clip_names)), match_counts, color='steelblue', alpha=0.7)
                
                # Add count labels on top of bars
                for i, (bar, count) in enumerate(zip(bars, match_counts)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(count), ha='center', va='bottom', fontweight='bold')
                
                plt.xlabel('Audio Clips')
                plt.ylabel('Number of Non-overlapping Matches')
                plt.title(f'Template Matches Summary for {template_file} (Threshold: {THRESHOLD})')
                plt.xticks(range(len(clip_names)), clip_names, rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                summary_plot_file = os.path.join(template_folder, f"matches_summary.png")
                plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                f.write(f"Summary bar graph for {template_file} saved to: {summary_plot_file}\n")
                f.write(f"Total clips processed for {template_file}: {len(clip_names)}\n")
                f.write(f"Total matches found for {template_file}: {sum(match_counts)}\n\n")

            # Update progress indicator
            template_count += 1
            progress_percent = (template_count / len(selected_templates)) * 100
            print(f"Progress: {template_count}/{len(selected_templates)} templates completed ({progress_percent:.1f}%)")

    f.write(f"All results saved to directory: {OUTPUT_DIR}\n")

# Print final summary to console
print(f"Template matching completed. Results saved to {OUTPUT_FILE}")
print(f"CSV matches saved to {CSV_OUTPUT_FILE}")
print(f"Total matches above threshold: {total_matches_count}")
print(f"Visualizations saved to directory: {OUTPUT_DIR}")