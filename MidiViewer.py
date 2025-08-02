import pygame
import os
import json
import numpy as np
import pretty_midi
import mir_eval
import io
from pydub import AudioSegment
import MidiFeatures

# --- User Settings ---
# [IMPORTANT] Please modify the paths below to match your environment.
PREDICTED_MIDI_PATH = "/Users/simhyeongju/AVAPT/EDA/_transcribed_MIDI/OnsetsAndFrames_2_5sec/"
GROUNDTRUTH_MIDI_PATH = "/Users/simhyeongju/AVAPT/data/pianovam/"
# -------------------------

pygame.init()
pygame.mixer.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MIDI Evaluation Viewer")

COLORS = {
    'background': (30, 30, 30), 'text': (240, 240, 240), 'button': (80, 80, 80),
    'button_hover': (110, 110, 110), 'header': (60, 60, 60),
    'tp_green': (50, 205, 50), 'fn_yellow_fill': (255, 215, 0), 'fp_red_outline': (255, 69, 0),
    'piano_roll_bg': (40, 40, 40), 'grid_line': (60, 60, 60), 'c_note_line': (80, 80, 80),
    'playback_line': (255, 0, 0),
    'gt_only_color': (0, 150, 255), 'pred_only_color': (255, 165, 0),
    'checkbox_border': (150, 150, 150), 'checkbox_checked': (200, 200, 200),
    'tooltip_bg': (250, 250, 220), 'tooltip_text': (10, 10, 10)
}

# --- Font Settings ---
FONT_FILE = 'NanumGothic-Regular.ttf'
try:
    FONT_MAIN = pygame.font.Font(FONT_FILE, 32)
    FONT_SMALL = pygame.font.Font(FONT_FILE, 22)
    FONT_TINY = pygame.font.Font(FONT_FILE, 18)
    print(f"✅ Font '{FONT_FILE}' loaded successfully.")
except Exception:
    print(f"⚠️ Warning: Font '{FONT_FILE}' not found. Using default font.")
    FONT_MAIN = pygame.font.Font(None, 36)
    FONT_SMALL = pygame.font.Font(None, 26)
    FONT_TINY = pygame.font.Font(None, 20)


class MidiEvaluatorApp:
    def __init__(self):
        self.running = True
        self.state = 'FILE_SELECT'
        self.clock = pygame.time.Clock()
        self.global_metadata = self._load_global_metadata()
        self.groundtruth_metadata = self._load_groundtruth_metadata()
        self.evaluation_metadata = self._load_evaluation_metadata()

        self.FILE_COLUMNS = {
            'index': {'label': '#', 'width': 60, 'key': 'index'},
            'piece': {'label': 'Piece', 'width': 500, 'key': 'piece'},
            'player': {'label': 'Player', 'width': 150, 'key': 'player'},
            'skill': {'label': 'Skill', 'width': 120, 'key': 'skill'},
            'split': {'label': 'Split', 'width': 100, 'key': 'split'},
            'name': {'label': 'Record Time', 'width': 280, 'key': 'name'},
            'p': {'label': 'P', 'width': 100, 'key': 'p'},
            'r': {'label': 'R', 'width': 100, 'key': 'r'},
            'f1': {'label': 'F1', 'width': 100, 'key': 'f1'}
        }
        self.SEGMENT_COLUMNS = {
            'index': {'label': '#', 'width': 100, 'key': 'index'},
            'num': {'label': 'Segment', 'width': 430, 'key': 'num'},
            'precision': {'label': 'Precision', 'width': 330, 'key': 'p'},
            'recall': {'label': 'Recall', 'width': 330, 'key': 'r'},
            'f1': {'label': 'F1-Score', 'width': 330, 'key': 'f1'}
        }
        
        self.features_info = {
            'Pitch Entropy': ('pitch_entropy', '{:.2f}', 'Pitch entropy across the segment.'),
            'Hi-Pitch': ('highest_pitch', '{}', 'Highest pitch (MIDI note) in the segment.'),
            'Lo-Pitch': ('lowest_pitch', '{}', 'Lowest pitch (MIDI note) in the segment.'),
            'Avg Pitch': ('average_pitch', '{:.2f}', 'Average pitch (MIDI note) in the segment.'),
            'Avg IOI': ('ioi_mean', '{:.2f}s', 'Average Inter-Onset-Interval (time between note starts).'),
            'IOI Std': ('ioi_std', '{:.2f}s', 'Standard deviation of IOIs.'),
            'Density': ('note_density', '{:.2f}', 'Average number of notes per second.'),
            'Avg Polyphony': ('average_polyphony', '{:.2f}', 'Average number of notes playing simultaneously.'),
            'Max Polyphony': ('max_polyphony', '{:.2f}', 'Maximum number of notes playing simultaneously.'),
            'Avg Len': ('average_note_length', '{:.2f}s', 'Average duration of a single note.'),
            'Max Len': ('max_note_length', '{:.2f}s', 'Duration of the longest note.'),
            'Min Len': ('min_note_length', '{:.2f}s', 'Duration of the shortest note.'),
            'Avg Int': ('interval_mean', '{:.2f}', 'Average pitch interval between consecutive notes.'),
        }

        self.file_list_sort_config = {'key': 'f1', 'ascending': True}
        self.segment_list_sort_config = {'key': 'num', 'ascending': True}
        self.audio_folders_data = self._load_and_sort_audio_folders()

        self.selected_file_info = {}
        self.selected_segment = None
        self.detail_data = {}; self.piano_roll_data = {}
        self.file_scroll_y = 0; self.segment_scroll_y = 0
        self.mouse_wheel_sensitivity = 30

        self.is_playing = False
        self.is_scrubbing = False
        self.current_playback_ms = 0.0
        self.play_start_tick = 0
        self.play_start_offset_ms = 0

        self.piano_roll_view_options = {
            'show_gt': True,
            'show_pred': True,
            'gt_checkbox_rect': pygame.Rect(0, 0, 0, 0),
            'pred_checkbox_rect': pygame.Rect(0, 0, 0, 0),
        }
        
        self.active_tooltip = None

    def _load_global_metadata(self):
        path = os.path.join(PREDICTED_MIDI_PATH, 'metadata.json')
        try:
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except FileNotFoundError: print(f"Error: Global metadata.json not found at '{path}'"); return None

    def _load_evaluation_metadata(self):
        path = os.path.join(PREDICTED_MIDI_PATH, 'evaluation.json')
        try:
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except FileNotFoundError: print(f"Warning: Evaluation metadata.json not found at '{path}'"); return None

    def _load_file_evaluation_metadata(self, filename):
        path = os.path.join(os.path.join(PREDICTED_MIDI_PATH, filename), 'evaluation.json')
        try:
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except FileNotFoundError: print(f"Warning: Evaluation metadata.json not found at '{path}'"); return None

    def _load_groundtruth_metadata(self):
        path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'metadata.json')
        processed_data = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for item in raw_data.values():
                    record_time = item.get("record_time")
                    if record_time:
                        processed_data[record_time] = item
            return processed_data
        except FileNotFoundError:
            print(f"Warning: Ground truth metadata.json not found at '{path}'")
            return {}

    def _load_and_sort_audio_folders(self):
        if not self.global_metadata or not self.evaluation_metadata:
            print("Error: Global metadata not loaded. Cannot create file list.")
            return []

        folder_data_list = []
        evaluation_results = self.evaluation_metadata.get('file_scores', {})
        print("Loading average scores from global metadata...")

        for name, eval_data in evaluation_results.items():
            base_name = name.replace('_16020Hz', '')
            gt_info = self.groundtruth_metadata.get(base_name, {})

            scores = {
                'p': eval_data.get('Precision', 0.0),
                'r': eval_data.get('Recall', 0.0),
                'f1': eval_data.get('F1-Score', 0.0)
            }

            skill = gt_info.get('P1_skill', 'N/A')
            if skill == "Intermediate": skill = "Inter"

            folder_data_list.append({
                'name': name,
                **scores,
                'piece': gt_info.get('piece', 'N/A')[:50],
                'split': gt_info.get('split', 'N/A'),
                'player': gt_info.get('P1_name', 'N/A'),
                'skill': skill
            })

        self._sort_list(folder_data_list, self.file_list_sort_config)
        print("Sorting complete.")
        return folder_data_list

    def _sort_list(self, data_list, config):
        try:
            data_list.sort(key=lambda x: x.get(config['key'], 0), reverse=not config['ascending'])
        except Exception as e: print(f"Sort failed: {e}")

    def _get_note_name(self, midi_number):
        if not (21 <= midi_number <= 108): return ""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1
        note_index = midi_number % 12
        return f"{note_names[note_index]}{octave}"

    def run(self):
        while self.running:
            self.handle_events(); self.update(); self.draw()
            self.clock.tick(60)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.MOUSEWHEEL:
                if self.state == 'FILE_SELECT': self.file_scroll_y -= event.y * self.mouse_wheel_sensitivity
                elif self.state == 'DETAIL_VIEW': self.segment_scroll_y -= event.y * self.mouse_wheel_sensitivity
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if event.button == 1:
                    if self.state == 'FILE_SELECT': self.handle_file_select_clicks(pos)
                    elif self.state == 'DETAIL_VIEW': self.handle_detail_view_clicks(pos)
                    elif self.state == 'PIANO_ROLL_VIEW': self.handle_piano_roll_clicks(pos)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.state == 'PIANO_ROLL_VIEW' and self.is_scrubbing:
                    self.is_scrubbing = False; self._set_playback_state(True)
            if event.type == pygame.MOUSEMOTION:
                if self.state == 'PIANO_ROLL_VIEW':
                    if self.is_scrubbing:
                        self.handle_piano_roll_scrub(event.pos)
                    self.check_feature_hover(event.pos)
                else:
                    self.active_tooltip = None

            if event.type == pygame.KEYDOWN:
                if self.state == 'PIANO_ROLL_VIEW':
                    if event.key == pygame.K_w:
                        self.piano_roll_view_options['show_gt'] = not self.piano_roll_view_options['show_gt']
                    elif event.key == pygame.K_s:
                        self.piano_roll_view_options['show_pred'] = not self.piano_roll_view_options['show_pred']
                    elif event.key == pygame.K_SPACE:
                        self._set_playback_state(not self.is_playing)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.change_segment(-1)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.change_segment(1)

                if event.key == pygame.K_ESCAPE:
                    self._go_back()

    def _go_back(self):
        if self.state == 'DETAIL_VIEW':
            self.state = 'FILE_SELECT'; self.detail_data = {}; self.segment_scroll_y = 0
        elif self.state == 'PIANO_ROLL_VIEW':
            self.state = 'DETAIL_VIEW'; self._set_playback_state(False); self.piano_roll_data = {}

    def update(self):
        if self.state == 'FILE_SELECT':
            max_scroll = max(0, len(self.audio_folders_data) * 40 - (SCREEN_HEIGHT - 100))
            self.file_scroll_y = max(0, min(self.file_scroll_y, max_scroll))
        elif self.state == 'DETAIL_VIEW' and self.detail_data.get('segment_list'):
            max_scroll = max(0, len(self.detail_data['segment_list']) * 35 - (SCREEN_HEIGHT - 135))
            self.segment_scroll_y = max(0, min(self.segment_scroll_y, max_scroll))

        if self.state == 'PIANO_ROLL_VIEW' and self.is_playing and not self.is_scrubbing:
            elapsed_ticks = pygame.time.get_ticks() - self.play_start_tick
            self.current_playback_ms = self.play_start_offset_ms + elapsed_ticks
            cut_length_ms = self.piano_roll_data.get('cut_length', 0) * 1000.0
            if cut_length_ms > 0 and self.current_playback_ms >= cut_length_ms:
                self.is_playing = False; pygame.mixer.music.stop()
                self.current_playback_ms = self.play_start_offset_ms

    def _set_playback_state(self, should_play):
        if should_play:
            if not self.is_playing:
                full_audio_slice = self.piano_roll_data.get('full_audio_slice')
                if full_audio_slice:
                    start_ms = int(self.current_playback_ms)
                    playback_slice = full_audio_slice[start_ms:]
                    sound_stream = io.BytesIO(); playback_slice.export(sound_stream, format="wav"); sound_stream.seek(0)
                    pygame.mixer.music.load(sound_stream); pygame.mixer.music.play()
                    self.play_start_tick = pygame.time.get_ticks(); self.play_start_offset_ms = self.current_playback_ms
                    self.is_playing = True
        else:
            if self.is_playing:
                self.is_playing = False; pygame.mixer.music.stop()

    def handle_table_header_click(self, pos, y_pos, columns, data_list, sort_config):
        x_offset = 50
        for col_name, col_info in columns.items():
            header_rect = pygame.Rect(x_offset, y_pos, col_info['width'], 40)
            if header_rect.collidepoint(pos):
                if sort_config['key'] == col_info['key']: sort_config['ascending'] = not sort_config['ascending']
                else: sort_config['key'] = col_info['key']; sort_config['ascending'] = True
                self._sort_list(data_list, sort_config); return True
            x_offset += col_info['width']
        return False

    def handle_file_select_clicks(self, pos):
        if self.handle_table_header_click(pos, 50, self.FILE_COLUMNS, self.audio_folders_data, self.file_list_sort_config): return
        list_rect = pygame.Rect(50, 90, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        if list_rect.collidepoint(pos):
            for i, folder_data in enumerate(self.audio_folders_data):
                item_rect = pygame.Rect(list_rect.x, list_rect.y + i * 40 - self.file_scroll_y, list_rect.width, 40)
                if item_rect.collidepoint(pos):
                    self.selected_file_info = folder_data; self.load_detail_data(); self.state = 'DETAIL_VIEW'
                    break

    def handle_detail_view_clicks(self, pos):
        if pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40).collidepoint(pos): self._go_back(); return
        if self.handle_table_header_click(pos, 95, self.SEGMENT_COLUMNS, self.detail_data.get('segment_list',[]), self.segment_list_sort_config): return
        list_rect = pygame.Rect(50, 135, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 135)
        if list_rect.collidepoint(pos) and self.detail_data.get('segment_list'):
            for i, seg_data in enumerate(self.detail_data['segment_list']):
                item_rect = pygame.Rect(list_rect.x, list_rect.y + i * 35 - self.segment_scroll_y, list_rect.width, 35)
                if item_rect.collidepoint(pos):
                    pygame.mixer.music.stop(); self.current_playback_ms = 0.0; self.is_playing = False; self.is_scrubbing = False
                    self.piano_roll_view_options['show_gt'] = True
                    self.piano_roll_view_options['show_pred'] = True
                    self.selected_segment = seg_data['num']; self.load_piano_roll_data(); self.state = 'PIANO_ROLL_VIEW'
                    break

    def handle_piano_roll_space_key(self):
        self._set_playback_state(not self.is_playing)

    def handle_piano_roll_clicks(self, pos):
        BACK_RECT = pygame.Rect(SCREEN_WIDTH - 130 - 10, 20, 120, 40)
        STOP_RECT = pygame.Rect(BACK_RECT.x - 120 - 10, 20, 120, 40)
        PLAY_RECT = pygame.Rect(STOP_RECT.x - 120 - 10, 20, 120, 40)
        NEXT_RECT = pygame.Rect(PLAY_RECT.x - 120 - 10, 20, 120, 40)
        PREV_RECT = pygame.Rect(NEXT_RECT.x - 120 - 10, 20, 120, 40)

        if BACK_RECT.collidepoint(pos): self._go_back(); return
        if PREV_RECT.collidepoint(pos): self.change_segment(-1); return
        if NEXT_RECT.collidepoint(pos): self.change_segment(1); return
        if PLAY_RECT.collidepoint(pos): self.handle_piano_roll_space_key()
        if STOP_RECT.collidepoint(pos):
            self.current_playback_ms = 0.0; self._set_playback_state(False)

        if self.piano_roll_view_options['gt_checkbox_rect'].collidepoint(pos):
            self.piano_roll_view_options['show_gt'] = not self.piano_roll_view_options['show_gt']
            return
        if self.piano_roll_view_options['pred_checkbox_rect'].collidepoint(pos):
            self.piano_roll_view_options['show_pred'] = not self.piano_roll_view_options['show_pred']
            return

        roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250)
        if roll_rect.collidepoint(pos):
            self.is_scrubbing = True; self._set_playback_state(False); self.handle_piano_roll_scrub(pos)

    def handle_piano_roll_scrub(self, pos):
        roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250)
        if self.piano_roll_data.get('cut_length'):
            time_ratio = (pos[0] - roll_rect.x) / roll_rect.width
            seek_time_sec = time_ratio * self.piano_roll_data.get('cut_length', 0)
            max_time_ms = self.piano_roll_data['cut_length'] * 1000.0
            self.current_playback_ms = max(0, min(seek_time_sec * 1000.0, max_time_ms))

    def change_segment(self, direction):
        segment_list = self.detail_data.get('segment_list')
        if not segment_list: return

        try:
            current_index = [i for i, seg in enumerate(segment_list) if seg['num'] == self.selected_segment][0]
        except IndexError:
            print(f"Error: Current segment {self.selected_segment} not found.")
            return

        new_index = current_index + direction

        if 0 <= new_index < len(segment_list):
            new_segment_num = segment_list[new_index]['num']
            self.selected_segment = new_segment_num
            self._set_playback_state(False)
            self.current_playback_ms = 0.0
            self.is_scrubbing = False
            self.load_piano_roll_data()
        else:
            print(f"No more segments in direction {direction}.")

    def load_detail_data(self):
        if not self.global_metadata:
            print("Error: Global metadata not loaded. Cannot load details.")
            self.detail_data = {}
            return

        try:
            selected_filename = self.selected_file_info['name']
            print(f"Loading detail data for {selected_filename}...")
            total_segment_count = self.global_metadata.get('items', {}).get(selected_filename, {}).get('num_midi_segments_created', 0)
            file_metadata = self._load_file_evaluation_metadata(selected_filename)
            eval_data_for_file = file_metadata.get('segments', {})

            segment_list = []
            for seg_num in range(1, total_segment_count + 1):
                scores = eval_data_for_file.get(str(seg_num), {})
                segment_list.append({
                    'num': seg_num,
                    'p': scores.get('Precision', 0.0),
                    'r': scores.get('Recall', 0.0),
                    'f1': scores.get('F1-Score', 0.0)
                })

            self.detail_data['segment_list'] = segment_list
            self._sort_list(self.detail_data['segment_list'], self.segment_list_sort_config)
        except Exception as e:
            print(f"Error loading detail data from global metadata: {e}")
            self.detail_data = {}

    def load_piano_roll_data(self):
        if not self.global_metadata:
            print("Warning: Global metadata not loaded. Cannot proceed.")
            self.piano_roll_data = {}
            return

        # Initialize
        pred_notes, gt_notes_in_segment = [], []
        matched_ref_indices, matched_est_indices = set(), set()
        frame_scores = {'p': 0.0, 'r': 0.0, 'f1': 0.0}
        frame_counts = {'TP': 0, 'FP': 0, 'TN': 0} # MODIFIED: FN -> TN
        audio_slice = None
        cut_length = 0
        gt_features, pred_features = None, None

        try:
            # --- Set up times and paths ---
            cut_length = self.global_metadata['midi_extractor_settings']['segment_length_sec']
            start_time = (self.selected_segment - 1) * cut_length
            end_time = start_time + cut_length
            selected_filename = self.selected_file_info['name']
            base_name = selected_filename.replace('_16020Hz', '')
            pred_path = os.path.join(PREDICTED_MIDI_PATH, selected_filename, f"{self.selected_segment}.mid")
            gt_path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'midi', f"{base_name}.mid")

            # --- Load Predicted MIDI ---
            try:
                if os.path.exists(pred_path):
                    pred_midi = pretty_midi.PrettyMIDI(pred_path)
                    if pred_midi.instruments and pred_midi.instruments[0].notes:
                        pred_notes = pred_midi.instruments[0].notes
                else: print(f"Warning: Predicted MIDI not found at '{pred_path}'")
            except Exception as e: print(f"Error loading predicted MIDI: {e}")

            # --- Load Ground Truth MIDI ---
            try:
                if os.path.exists(gt_path):
                    gt_midi = pretty_midi.PrettyMIDI(gt_path)
                    for inst in gt_midi.instruments:
                        for note in inst.notes:
                            if note.start < end_time and note.end > start_time:
                                new_note = pretty_midi.Note(
                                    velocity=note.velocity, pitch=note.pitch,
                                    start=max(0, note.start - start_time),
                                    end=min(cut_length, note.end - start_time)
                                )
                                if new_note.end > new_note.start:
                                    gt_notes_in_segment.append(new_note)
                else: print(f"Warning: Ground truth MIDI not found at '{gt_path}'")
            except Exception as e: print(f"Error loading ground truth MIDI: {e}")

            # --- Extract MIDI Features ---
            if gt_notes_in_segment:
                temp_gt_midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)
                instrument.notes.extend(gt_notes_in_segment)
                temp_gt_midi.instruments.append(instrument)
                temp_gt_path = "temp_gt_segment_features.mid"
                temp_gt_midi.write(temp_gt_path)
                gt_features_obj = MidiFeatures.MidiFeatures(temp_gt_path)
                if gt_features_obj.available:
                    gt_features = gt_features_obj.numeric_features
                os.remove(temp_gt_path)

            if os.path.exists(pred_path) and pred_notes:
                pred_features_obj = MidiFeatures.MidiFeatures(pred_path)
                if pred_features_obj.available:
                    pred_features = pred_features_obj.numeric_features

            # --- Note Matching (mir_eval) ---
            if gt_notes_in_segment and pred_notes:
                ref_intervals_note = np.array([[n.start, n.end] for n in gt_notes_in_segment])
                ref_pitches_note = np.array([n.pitch for n in gt_notes_in_segment])
                est_intervals_note = np.array([[n.start, n.end] for n in pred_notes])
                est_pitches_note = np.array([n.pitch for n in pred_notes])
                if ref_intervals_note.size > 0 and est_intervals_note.size > 0:
                    matching = mir_eval.transcription.match_notes(
                        ref_intervals_note, ref_pitches_note, est_intervals_note, est_pitches_note
                    )
                    matched_ref_indices = {m[0] for m in matching}
                    matched_est_indices = {m[1] for m in matching}

            # --- Frame-based Score Calculation ---
            frame_len_sec = 0.01
            frame_times = np.arange(0, cut_length, frame_len_sec)
            total_tp, total_fp, total_tn = 0, 0, 0 # MODIFIED: fn -> tn
            for time in frame_times:
                true_pitches = {note.pitch for note in gt_notes_in_segment if note.start <= time < note.end}
                pred_pitches = {note.pitch for note in pred_notes if note.start <= time < note.end}
                total_tp += len(true_pitches.intersection(pred_pitches))
                total_fp += len(pred_pitches.difference(true_pitches))
                total_tn += len(true_pitches.difference(pred_pitches)) # MODIFIED: fn -> tn

            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r = total_tp / (total_tp + total_tn) if (total_tp + total_tn) > 0 else 0.0 # MODIFIED: fn -> tn
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            frame_scores = {'p': p, 'r': r, 'f1': f1}
            frame_counts = {'TP': total_tp, 'FP': total_fp, 'TN': total_tn} # MODIFIED: FN -> TN

            # --- Load and Slice Audio ---
            audio_path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'audio', f"{selected_filename}.wav")
            if os.path.exists(audio_path):
                full_audio = AudioSegment.from_wav(audio_path)
                audio_slice = full_audio[start_time * 1000 : end_time * 1000]
            else:
                print(f"Warning: Audio file not found at '{audio_path}'")

        except Exception as e:
            print(f"An unexpected error occurred in load_piano_roll_data: {e}")

        finally:
            self.piano_roll_data = {
                'pred_notes': pred_notes, 'gt_notes': gt_notes_in_segment,
                'matched_ref_indices': matched_ref_indices, 'matched_est_indices': matched_est_indices,
                'cut_length': cut_length, 'frame_scores': frame_scores, 'frame_counts': frame_counts,
                'full_audio_slice': audio_slice, 'gt_features': gt_features, 'pred_features': pred_features
            }

    def draw(self):
        screen.fill(COLORS['background'])
        if self.state == 'FILE_SELECT': self.draw_file_select_screen()
        elif self.state == 'DETAIL_VIEW': self.draw_detail_view_screen()
        elif self.state == 'PIANO_ROLL_VIEW': self.draw_piano_roll_screen()
        
        self._draw_tooltip()
        pygame.display.flip()

    def _draw_table_header(self, y_pos, columns, sort_config):
        header_rect = pygame.Rect(50, y_pos, SCREEN_WIDTH - 100, 40)
        pygame.draw.rect(screen, COLORS['header'], header_rect)
        x_offset = 50
        for col_name, col_info in columns.items():
            label = col_info['label']
            if col_info['key'] == sort_config['key']:
                label += " (asc)" if sort_config['ascending'] else " (desc)"
            text_surf = FONT_SMALL.render(label, True, COLORS['text'])
            screen.blit(text_surf, (x_offset + 5, y_pos + 10))
            x_offset += col_info['width']

    def draw_file_select_screen(self):
        screen.blit(FONT_MAIN.render("Select an Audio File", True, COLORS['text']), (50, 10))
        self._draw_table_header(50, self.FILE_COLUMNS, self.file_list_sort_config)
        list_rect = pygame.Rect(50, 90, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        for i, folder_data in enumerate(self.audio_folders_data):
            y_pos = list_rect.y + i * 40 - self.file_scroll_y
            if list_rect.top <= y_pos < list_rect.bottom - 40:
                item_rect = pygame.Rect(list_rect.x, y_pos, list_rect.width, 40)
                if item_rect.collidepoint(pygame.mouse.get_pos()): pygame.draw.rect(screen, COLORS['button_hover'], item_rect)
                x_offset = 50
                record_time_full = folder_data['name'].replace('_16020Hz', '')

                try:
                    segs = record_time_full.split('_')
                    display_name = segs[0][2:] + ' ' + segs[1].replace('-',':')
                except IndexError:
                    display_name = record_time_full
                col_data = {'index': str(i + 1), 'name': display_name, 'p': f"{folder_data['p']:.4f}", 'r': f"{folder_data['r']:.4f}", 'f1': f"{folder_data['f1']:.4f}", 'piece': folder_data.get('piece', 'N/A'), 'player': folder_data.get('player', 'N/A'), 'skill': folder_data.get('skill', 'N/A'), 'split': folder_data.get('split', 'N/A')}
                for col_name, col_info in self.FILE_COLUMNS.items():
                    text_surf = FONT_SMALL.render(str(col_data.get(col_info['key'], '')), True, COLORS['text'])
                    screen.blit(text_surf, (x_offset + 5, item_rect.y + 10))
                    x_offset += col_info['width']

    def draw_detail_view_screen(self):
        back_button_rect = pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40)
        if back_button_rect.collidepoint(pygame.mouse.get_pos()): pygame.draw.rect(screen, COLORS['button_hover'], back_button_rect, border_radius=5)
        else: pygame.draw.rect(screen, COLORS['button'], back_button_rect, border_radius=5)
        screen.blit(FONT_SMALL.render("<< Back", True, COLORS['text']), (back_button_rect.x + 30, back_button_rect.y + 10))
        info = self.selected_file_info
        title_surf = FONT_MAIN.render(f"Piece: {info.get('piece', 'N/A')}", True, COLORS['text'])
        subtitle1_surf = FONT_SMALL.render(f"Player: {info.get('player', 'N/A')} ({info.get('skill', 'N/A')})   |   Split: {info.get('split', 'N/A')}", True, COLORS['text'])
        subtitle2_surf = FONT_TINY.render(f"File: {info.get('name', '')}", True, (180, 180, 180))
        screen.blit(title_surf, (50, 10)); screen.blit(subtitle1_surf, (50, 45)); screen.blit(subtitle2_surf, (50, 70))
        self._draw_table_header(95, self.SEGMENT_COLUMNS, self.segment_list_sort_config)
        list_rect = pygame.Rect(50, 135, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 135)
        if self.detail_data.get('segment_list'):
            for i, seg_data in enumerate(self.detail_data['segment_list']):
                y_pos = list_rect.y + i * 35 - self.segment_scroll_y
                if list_rect.top <= y_pos < list_rect.bottom - 35:
                    item_rect = pygame.Rect(list_rect.x, y_pos, list_rect.width, 35)
                    if item_rect.collidepoint(pygame.mouse.get_pos()): pygame.draw.rect(screen, COLORS['button_hover'], item_rect)
                    x_offset = 50
                    col_data = {'index': str(i + 1), 'num': str(seg_data['num']), 'p': f"{seg_data['p']:.4f}", 'r': f"{seg_data['r']:.4f}", 'f1': f"{seg_data['f1']:.4f}"}
                    for col_name, col_info in self.SEGMENT_COLUMNS.items():
                        text_surf = FONT_SMALL.render(str(col_data.get(col_info['key'], '')), True, COLORS['text'])
                        screen.blit(text_surf, (x_offset + 10, item_rect.y + 8))
                        x_offset += col_info['width']

    def check_feature_hover(self, pos):
        table_rect = pygame.Rect(250, SCREEN_HEIGHT - 150 + 20, SCREEN_WIDTH - 300, 90)
        num_cols = len(self.features_info)
        col_width = table_rect.width / num_cols
        x_offset = table_rect.x

        for label, (key, fmt, desc) in self.features_info.items():
            cell_rect = pygame.Rect(x_offset, table_rect.y, col_width, table_rect.height)
            if cell_rect.collidepoint(pos):
                self.active_tooltip = {'text': desc, 'pos': pos}
                return
            x_offset += col_width
        self.active_tooltip = None

    def _draw_tooltip(self):
        if not self.active_tooltip: return

        text_surf = FONT_TINY.render(self.active_tooltip['text'], True, COLORS['tooltip_text'])
        padding = 8
        bg_rect = text_surf.get_rect().inflate(padding * 2, padding * 2)
        
        mouse_pos = self.active_tooltip['pos']
        bg_rect.bottomleft = (mouse_pos[0] + 15, mouse_pos[1] - 5)
        
        if bg_rect.right > SCREEN_WIDTH: bg_rect.right = SCREEN_WIDTH - 5
        if bg_rect.left < 0: bg_rect.left = 5
        if bg_rect.top < 0: bg_rect.top = 5
        
        pygame.draw.rect(screen, COLORS['tooltip_bg'], bg_rect, border_radius=5)
        screen.blit(text_surf, (bg_rect.x + padding, bg_rect.y + padding))

    def _draw_legend_and_controls(self, start_x, start_y):
        y_offset = 0
        show_gt = self.piano_roll_view_options['show_gt']
        show_pred = self.piano_roll_view_options['show_pred']
        checkbox_size = 20

        gt_rect = pygame.Rect(start_x, start_y + y_offset, checkbox_size, checkbox_size)
        self.piano_roll_view_options['gt_checkbox_rect'] = gt_rect
        pygame.draw.rect(screen, COLORS['checkbox_border'], gt_rect, 2, border_radius=3)
        screen.blit(FONT_SMALL.render("GT MIDI (W)", True, COLORS['text']), (gt_rect.x + 30, gt_rect.y - 2))
        if show_gt: pygame.draw.rect(screen, COLORS['checkbox_checked'], gt_rect.inflate(-6, -6), border_radius=3)
        y_offset += 35

        pred_rect = pygame.Rect(start_x, start_y + y_offset, checkbox_size, checkbox_size)
        self.piano_roll_view_options['pred_checkbox_rect'] = pred_rect
        pygame.draw.rect(screen, COLORS['checkbox_border'], pred_rect, 2, border_radius=3)
        screen.blit(FONT_SMALL.render("Pred MIDI (S)", True, COLORS['text']), (pred_rect.x + 30, pred_rect.y - 2))
        if show_pred: pygame.draw.rect(screen, COLORS['checkbox_checked'], pred_rect.inflate(-6, -6), border_radius=3)
        y_offset += 55

        legend_items = []
        if show_gt and show_pred:
            legend_items = [("Correct (TP)", COLORS['tp_green'], 'fill'), ("Extra Note (FP)", COLORS['fp_red_outline'], 'outline'), ("Missed Note (TN)", COLORS['fn_yellow_fill'], 'fill')]
        elif show_gt:
            legend_items = [("Ground Truth Note", COLORS['gt_only_color'], 'fill')]
        elif show_pred:
            legend_items = [("Predicted Note", COLORS['pred_only_color'], 'fill')]

        for label, color, style in legend_items:
            swatch_rect = pygame.Rect(start_x, start_y + y_offset, 20, 15)
            if style == 'fill': pygame.draw.rect(screen, color, swatch_rect, border_radius=3)
            else: pygame.draw.rect(screen, color, swatch_rect, 2, border_radius=3)
            text_surf = FONT_SMALL.render(label, True, COLORS['text'])
            screen.blit(text_surf, (start_x + 30, start_y + y_offset - 2))
            y_offset += 30

    def _draw_midi_features_table(self, rect):
        gt_f = self.piano_roll_data.get('gt_features')
        if not gt_f: return

        pygame.draw.rect(screen, COLORS['header'], rect, border_radius=5)
        pygame.draw.rect(screen, COLORS['grid_line'], rect, 1, border_radius=5)

        title_surf = FONT_SMALL.render("Ground Truth MIDI Features", True, COLORS['text'])
        screen.blit(title_surf, (rect.x + 10, rect.y + 5))
        
        num_cols = len(self.features_info)
        col_width = rect.width / num_cols
        table_y = rect.y + 35
        cell_height = rect.height - 38
        x_offset = rect.x

        for label, (key, fmt, desc) in self.features_info.items():
            cell_rect = pygame.Rect(x_offset, table_y, col_width, cell_height)

            label_surf = FONT_TINY.render(label, True, COLORS['gt_only_color'])
            label_rect = label_surf.get_rect(centerx=cell_rect.centerx, top=cell_rect.top + 8)
            screen.blit(label_surf, label_rect)
            
            val_str = "N/A"
            if key in gt_f:
                val = gt_f[key]
                try: val_str = fmt.format(val)
                except (ValueError, TypeError): val_str = str(val)

            value_surf = FONT_SMALL.render(val_str, True, COLORS['text'])
            value_rect = value_surf.get_rect(centerx=cell_rect.centerx, bottom=cell_rect.bottom - 8)
            screen.blit(value_surf, value_rect)

            x_offset += col_width
            if x_offset < rect.right:
                pygame.draw.line(screen, COLORS['grid_line'], (x_offset, table_y), (x_offset, table_y + cell_height))

    def _draw_piano_roll_stats(self):
        if not self.piano_roll_data: return
        scores = self.piano_roll_data.get('frame_scores', {})
        counts = self.piano_roll_data.get('frame_counts', {})
        score_text = f"Frame Precision: {scores.get('p', 0):.4f}   Recall: {scores.get('r', 0):.4f}   F1-Score: {scores.get('f1', 0):.4f}"
        count_text = f"Frame TP: {counts.get('TP', 0)}   Extra Notes (FP): {counts.get('FP', 0)}   Missed Notes (TN): {counts.get('TN', 0)}" # MODIFIED: FN -> TN
        score_surf = FONT_SMALL.render(score_text, True, COLORS['text'])
        count_surf = FONT_SMALL.render(count_text, True, COLORS['text'])
        screen.blit(score_surf, (50, 45)); screen.blit(count_surf, (50, 70))

    def draw_piano_roll_screen(self):
        BACK_RECT = pygame.Rect(SCREEN_WIDTH - 130 - 10, 20, 120, 40)
        STOP_RECT = pygame.Rect(BACK_RECT.x - 120 - 10, 20, 120, 40)
        PLAY_RECT = pygame.Rect(STOP_RECT.x - 120 - 10, 20, 120, 40)
        NEXT_RECT = pygame.Rect(PLAY_RECT.x - 120 - 10, 20, 120, 40)
        PREV_RECT = pygame.Rect(NEXT_RECT.x - 120 - 10, 20, 120, 40)

        mouse_pos = pygame.mouse.get_pos()
        def draw_button(rect, text):
            color = COLORS['button_hover'] if rect.collidepoint(mouse_pos) else COLORS['button']
            pygame.draw.rect(screen, color, rect, border_radius=5)
            text_surf = FONT_SMALL.render(text, True, COLORS['text'])
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

        draw_button(PREV_RECT, "<< Prev"); draw_button(NEXT_RECT, "Next >>")
        draw_button(PLAY_RECT, "Pause" if self.is_playing else "Play"); draw_button(STOP_RECT, "Stop")
        draw_button(BACK_RECT, "<< Back")

        info = self.selected_file_info
        title_surf = FONT_MAIN.render(f"Piece: {info.get('piece', 'N/A')} - Segment {self.selected_segment}", True, COLORS['text'])
        screen.blit(title_surf, (50, 15))

        self._draw_piano_roll_stats()

        roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250)
        pygame.draw.rect(screen, COLORS['piano_roll_bg'], roll_rect)
        if not self.piano_roll_data: return
        min_pitch, max_pitch = 21, 108; pitch_span = max_pitch - min_pitch
        for pitch in range(min_pitch, max_pitch + 1):
            y = roll_rect.bottom - ((pitch - min_pitch) / pitch_span) * roll_rect.height
            color = COLORS['c_note_line'] if self._get_note_name(pitch).startswith('C') else COLORS['grid_line']
            pygame.draw.line(screen, color, (roll_rect.left, y), (roll_rect.right, y), 1)
        for i in range(1, 9):
            pitch = 12 * (i + 1)
            if min_pitch <= pitch <= max_pitch:
                y = roll_rect.bottom - ((pitch - min_pitch) / pitch_span) * roll_rect.height
                screen.blit(FONT_SMALL.render(f"C{i}", True, COLORS['text']), (roll_rect.left - 40, y - 10))

        note_height = roll_rect.height / pitch_span if pitch_span > 0 else 0
        time_span = self.piano_roll_data.get('cut_length', 0)
        show_gt = self.piano_roll_view_options['show_gt']
        show_pred = self.piano_roll_view_options['show_pred']

        if time_span > 0:
            if show_gt and show_pred:
                for gt_idx, gt_note in enumerate(self.piano_roll_data['gt_notes']):
                    x = roll_rect.x + (gt_note.start / time_span) * roll_rect.width
                    y = roll_rect.bottom - ((gt_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height
                    width = max(1, (gt_note.end - gt_note.start) / time_span * roll_rect.width)
                    color = COLORS['tp_green'] if gt_idx in self.piano_roll_data['matched_ref_indices'] else COLORS['fn_yellow_fill']
                    pygame.draw.rect(screen, color, (x, y, width, note_height), border_radius=2)
                for pred_idx, pred_note in enumerate(self.piano_roll_data['pred_notes']):
                    if pred_idx not in self.piano_roll_data['matched_est_indices']:
                        x = roll_rect.x + (pred_note.start / time_span) * roll_rect.width
                        y = roll_rect.bottom - ((pred_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height
                        width = max(1, (pred_note.end - pred_note.start) / time_span * roll_rect.width)
                        pygame.draw.rect(screen, COLORS['fp_red_outline'], (x, y, width, note_height), 2, border_radius=2)
            elif show_gt:
                for gt_note in self.piano_roll_data['gt_notes']:
                    x, y = roll_rect.x + (gt_note.start / time_span) * roll_rect.width, roll_rect.bottom - ((gt_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height
                    width = max(1, (gt_note.end - gt_note.start) / time_span * roll_rect.width)
                    pygame.draw.rect(screen, COLORS['gt_only_color'], (x, y, width, note_height), border_radius=2)
            elif show_pred:
                for pred_note in self.piano_roll_data['pred_notes']:
                    x, y = roll_rect.x + (pred_note.start / time_span) * roll_rect.width, roll_rect.bottom - ((pred_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height
                    width = max(1, (pred_note.end - pred_note.start) / time_span * roll_rect.width)
                    pygame.draw.rect(screen, COLORS['pred_only_color'], (x, y, width, note_height), border_radius=2)

        if time_span > 0:
            line_x = roll_rect.x + (self.current_playback_ms / (time_span * 1000.0)) * roll_rect.width
            if roll_rect.left <= line_x <= roll_rect.right:
                pygame.draw.line(screen, COLORS['playback_line'], (line_x, roll_rect.top), (line_x, roll_rect.bottom), 2)

        # --- Draw Controls & Features ---
        self._draw_legend_and_controls(30, 150)
        features_rect = pygame.Rect(250, roll_rect.bottom + 20, roll_rect.width, 90)
        self._draw_midi_features_table(features_rect)


if __name__ == '__main__':
    if not os.path.exists(PREDICTED_MIDI_PATH) or not os.path.exists(GROUNDTRUTH_MIDI_PATH):
        print("="*60)
        print("! ! ! CONFIGURATION ERROR ! ! !")
        print(f"Cannot find PREDICTED_MIDI_PATH or GROUNDTRUTH_MIDI_PATH.")
        print("Please correctly modify the path settings at the top of the script.")
        print(f"Predicted MIDI Path: {PREDICTED_MIDI_PATH}")
        print(f"Ground Truth MIDI Path: {GROUNDTRUTH_MIDI_PATH}")
        print("="*60)
    else:
        app = MidiEvaluatorApp()
        app.run()