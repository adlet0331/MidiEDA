import json
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class NumpyFloatEncoder(json.JSONEncoder):
    """Encoder class to save numpy float types to json."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(round(obj, 4))
        return json.JSONEncoder.default(self, obj)

class MetadataAnalyzer:
    """
    Analyzes music transcription evaluation metadata and generates statistics
    and graphs for each file.
    """
    def __init__(self, base_folder: str):
        """
        Initializes the analyzer class.
        Args:
            base_folder (str): The base path where the audio file folders to be analyzed are located.
        """
        self.base_folder = base_folder
        # Removed font setting logic.

    def _plot_distribution(self, metric_name, distribution_data, filename, output_folder, save_plots):
        """Generates and displays/saves a distribution graph for a single metric."""
        fig, ax = plt.subplots(figsize=(16, 9))
        labels, counts = list(distribution_data.keys()), [d['count'] for d in distribution_data.values()]

        ax.bar(labels, counts, color='darkcyan', alpha=0.9, edgecolor='black', width=0.8)
        # --- Titles and labels changed to English ---
        ax.set_title(f"'{metric_name}' Score Distribution\n(File: {filename})", fontsize=20, pad=20)
        ax.set_xlabel('Score Range', fontsize=15, labelpad=15)
        ax.set_ylabel('Segment Count', fontsize=15, labelpad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        for bar in ax.patches:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), int(bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        if counts:
            ax.set_ylim(0, max(counts) * 1.15)

        if save_plots:
            plot_output_path = os.path.join(output_folder, f"{metric_name}_Distribution.png")
            plt.savefig(plot_output_path, dpi=150, bbox_inches='tight')
            # --- Print statement changed to English ---
            print(f"  - Plot saved to: '{plot_output_path}'")
            plt.close(fig)
        else:
            plt.show()

    def _process_file(self, filename, data, output_folder, save_plots):
        """Analyzes a single evaluation.json data and generates outputs."""
        # --- Print statements changed to English ---
        print("-" * 60)
        print(f"Processing file: {filename}")

        os.makedirs(output_folder, exist_ok=True)

        segments = data.get('segments', {})
        if not segments:
            print("  No segment data found for this file.")
            return

        scores = {'Precision': [], 'Recall': [], 'F1-Score': []}
        for seg_num, seg_scores in segments.items():
            if seg_scores:
                for metric in scores.keys():
                    if metric in seg_scores:
                        scores[metric].append((seg_scores[metric], seg_num))

        num_segments = len(scores['F1-Score'])
        print(f"  Analyzing {num_segments} segments...")

        statistics = {}
        bins = np.arange(0.8, 1.01, 0.02)
        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]

        for metric, tuples in scores.items():
            scores_array = np.array([t[0] for t in tuples if t[0] is not None])
            if len(scores_array) == 0:
                continue

            stats_data = {
                'mean': np.mean(scores_array), 'std_dev': np.std(scores_array), 'median': np.median(scores_array),
                'max_score': {'value': np.max(scores_array), 'segment': f"segment {tuples[np.argmax(scores_array)][1]}"},
                'min_score': {'value': np.min(scores_array), 'segment': f"segment {tuples[np.argmin(scores_array)][1]}"}
            }

            dist_data = {label: {"count": 0, "segments": []} for label in bin_labels}
            for score, seg_num in tuples:
                if score is not None and score >= 0.8:
                    idx = np.digitize(score, bins) - 1
                    if score == 1.0: idx = len(bin_labels) - 1
                    if 0 <= idx < len(bin_labels):
                        dist_data[bin_labels[idx]]["count"] += 1
                        dist_data[bin_labels[idx]]["segments"].append(f"segment {seg_num}")

            stats_data['distribution_0.8_to_1.0'] = dist_data
            statistics[metric] = stats_data

            self._plot_distribution(metric, dist_data, filename, output_folder, save_plots)

        output_data = {
            'info': f"Statistics for {filename}", 'model': data.get('model', 'Unknown'),
            'source_file': filename, 'num_segments_analyzed': num_segments, 'statistics': statistics
        }
        output_json_path = os.path.join(output_folder, 'statistics_metadata.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, cls=NumpyFloatEncoder, ensure_ascii=False)
        print(f"  - Statistics saved to: '{output_json_path}'")

    def run_analysis_by_folder(self, save_plots: bool = True):
        """
        Finds and runs analysis on 'evaluation.json' files within each subfolder of the base folder.
        """
        # --- Print statements changed to English ---
        print(f"Scanning for 'evaluation.json' files in '{self.base_folder}'...")

        found_files = 0
        for item_name in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item_name)

            if os.path.isdir(item_path):
                eval_json_path = os.path.join(item_path, 'evaluation.json')

                if os.path.exists(eval_json_path):
                    found_files += 1
                    try:
                        with open(eval_json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        filename = data.get('audio_filename', item_name)
                        output_folder = item_path

                        self._process_file(filename, data, output_folder, save_plots)

                    except json.JSONDecodeError:
                        print(f"Error: '{eval_json_path}' is not a valid JSON file.")
                    except Exception as e:
                        print(f"An exception occurred while processing '{item_name}': {e}")

        if found_files == 0:
            print("\nCould not find any 'evaluation.json' files to analyze.")
        else:
            print("-" * 60)
            print(f"\nFinished analysis for a total of {found_files} files!")