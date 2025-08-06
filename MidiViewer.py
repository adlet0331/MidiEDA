import pygame
import os
import json
import numpy as np
import pretty_midi
import mir_eval
import io
from pydub import AudioSegment
from MidiFeatures import MidiFeatures

# --- User Settings ---
# [IMPORTANT] Please modify the paths below to match your environment.
PREDICTED_MIDI_PATH = "/Users/simhyeongju/AVAPT/EDA/_transcribed_MIDI/OnsetsAndFrames_2_5sec/"
GROUNDTRUTH_MIDI_PATH = "/Users/simhyeongju/AVAPT/data/pianovam/"
SCORE_SCALE_CONSTANT = 10
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
    'tooltip_bg': (250, 250, 220), 'tooltip_text': (10, 10, 10),
    'popup_overlay': (0, 0, 0, 180),
    'regression_line': (255, 105, 180)
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
        self.groundtruth_features_metadata = self._load_groundtruth_features_metadata()

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
        self.SEGMENT_COLUMNS_BASE = {
            'index': {'label': '#', 'width': 60, 'key': 'index'},
            'num': {'label': 'Segment', 'width': 120, 'key': 'num'},
            'precision': {'label': 'Precision', 'width': 120, 'key': 'p'},
            'recall': {'label': 'Recall', 'width': 120, 'key': 'r'},
            'f1': {'label': 'F1-Score', 'width': 120, 'key': 'f1'}
        }
        self.dynamic_segment_columns = self.SEGMENT_COLUMNS_BASE.copy()
        
        self.features_info = {
            'Pitch Entropy': ('pitch_entropy', '{:.2f}', 'Pitch entropy across the segment.'),
            'Hi-Pitch': ('highest_pitch', '{}', 'Highest pitch (MIDI note) in the segment.'),
            'Lo-Pitch': ('lowest_pitch', '{}', 'Lowest pitch (MIDI note) in the segment.'),
            'Pitch Avg': ('average_pitch', '{:.2f}', 'Average pitch (MIDI note) in the segment.'),
            'IOI Avg': ('ioi_mean', '{:.2f}s', 'Average Inter-Onset-Interval (time between note starts).'),
            'IOI Entropy': ('ioi_entropy', '{:.2f}', 'Entropy of IOIs.'),
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

        self.is_playing = False; self.is_scrubbing = False
        self.current_playback_ms = 0.0; self.play_start_tick = 0; self.play_start_offset_ms = 0

        self.piano_roll_view_options = { 'show_gt': True, 'show_pred': True, 'gt_checkbox_rect': pygame.Rect(0, 0, 0, 0), 'pred_checkbox_rect': pygame.Rect(0, 0, 0, 0) }
        
        self.show_feature_popup = False
        self.selected_features = set([v[0] for v in list(self.features_info.values())[:5]])
        self.feature_select_button_rect = pygame.Rect(SCREEN_WIDTH - 320, 20, 160, 40)
        self.detail_corr_button_rect = pygame.Rect(self.feature_select_button_rect.x - 170, 20, 160, 40)

        popup_w, popup_h = 600, 500
        self.feature_popup_rect = pygame.Rect(SCREEN_WIDTH // 2 - popup_w // 2, SCREEN_HEIGHT // 2 - popup_h // 2, popup_w, popup_h)
        self.feature_popup_close_button_rect = pygame.Rect(self.feature_popup_rect.right - 120, self.feature_popup_rect.bottom - 55, 100, 40)
        self.feature_checkbox_rects = {}
        self.active_tooltip = None

        self.correlation_data = {}
        self.selected_correlation_feature = None
        self.selected_score_metric = 'f1'
        self.correlation_feature_buttons = {}
        self.correlation_metric_buttons = {}
        
        self.correlation_normalize = False
        self.show_regression_line = False
        self.correlation_plot_type = 'scatter'
        self.correlation_controls = {}
        
        # --- DropDown Menu for Plot Types ---
        self.plot_type_dropdown_open = False
        self.plot_types = {
            'scatter': 'Scatter (1)',
            'hist_feature': 'Feat Hist (2)',
            'hist_score': 'Score Hist (3)'
        }
        self.dropdown_rects = {} # Stores rects for the dropdown and its options

        self.global_correlation_data = {}
        self.selected_global_correlation_feature = None
        self.selected_global_score_metric = 'f1'
        self.global_correlation_feature_buttons = {}
        self.global_correlation_metric_buttons = {}

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

    def _load_groundtruth_features_metadata(self):
        path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'midi', 'features_metadata.json')
        print(f"Loading ground truth features from '{path}'...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print("✅ Ground truth features loaded successfully.")
                return data
        except FileNotFoundError:
            print(f"⚠️ Warning: Ground truth features_metadata.json not found at '{path}'")
            return {}

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
                    if record_time: processed_data[record_time] = item
            return processed_data
        except FileNotFoundError: print(f"Warning: Ground truth metadata.json not found at '{path}'"); return {}

    def _load_and_sort_audio_folders(self):
        if not self.global_metadata or not self.evaluation_metadata: print("Error: Global metadata not loaded. Cannot create file list."); return []
        folder_data_list = []
        evaluation_results = self.evaluation_metadata.get('file_scores', {})
        print("Loading average scores from global metadata...")
        for name, eval_data in evaluation_results.items():
            base_name = name.replace('_16020Hz', ''); gt_info = self.groundtruth_metadata.get(base_name, {})
            scores = {'p': eval_data.get('Precision', 0.0), 'r': eval_data.get('Recall', 0.0), 'f1': eval_data.get('F1-Score', 0.0)}
            skill = gt_info.get('P1_skill', 'N/A');
            if skill == "Intermediate": skill = "Inter"
            folder_data_list.append({'name': name, **scores, 'piece': gt_info.get('piece', 'N/A')[:50], 'split': gt_info.get('split', 'N/A'), 'player': gt_info.get('P1_name', 'N/A'), 'skill': skill})
        self._sort_list(folder_data_list, self.file_list_sort_config); print("Sorting complete."); return folder_data_list

    def _sort_list(self, data_list, config):
        key = config['key']
        is_feature = key not in self.FILE_COLUMNS and key not in ['p', 'r', 'f1', 'num', 'index']
        try:
            if is_feature: data_list.sort(key=lambda x: x.get('features', {}).get(key, -1), reverse=not config['ascending'])
            else: data_list.sort(key=lambda x: x.get(key, 0), reverse=not config['ascending'])
        except Exception as e: print(f"Sort failed: {e}")

    def _get_note_name(self, midi_number):
        if not (21 <= midi_number <= 108): return ""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1; note_index = midi_number % 12
        return f"{note_names[note_index]}{octave}"

    def run(self):
        while self.running: self.handle_events(); self.update(); self.draw(); self.clock.tick(60)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.MOUSEWHEEL:
                if self.state == 'FILE_SELECT': self.file_scroll_y -= event.y * self.mouse_wheel_sensitivity
                elif self.state == 'DETAIL_VIEW' and not self.show_feature_popup: self.segment_scroll_y -= event.y * self.mouse_wheel_sensitivity
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                if self.state == 'FILE_SELECT': self.handle_file_select_clicks(pos)
                elif self.state == 'DETAIL_VIEW': self.handle_detail_view_clicks(pos)
                elif self.state == 'PIANO_ROLL_VIEW': self.handle_piano_roll_clicks(pos)
                elif self.state in ['CORRELATION_VIEW', 'GLOBAL_CORRELATION_VIEW']:
                    self.handle_correlation_view_clicks(pos)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.state == 'PIANO_ROLL_VIEW' and self.is_scrubbing: self.is_scrubbing = False; self._set_playback_state(True)
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                self.active_tooltip = None
                if self.state == 'PIANO_ROLL_VIEW':
                    if self.is_scrubbing: self.handle_piano_roll_scrub(mouse_pos)
                    self.check_feature_hover(mouse_pos)
            if event.type == pygame.KEYDOWN:
                if self.state == 'PIANO_ROLL_VIEW':
                    if event.key == pygame.K_w: self.piano_roll_view_options['show_gt'] = not self.piano_roll_view_options['show_gt']
                    elif event.key == pygame.K_s: self.piano_roll_view_options['show_pred'] = not self.piano_roll_view_options['show_pred']
                    elif event.key == pygame.K_SPACE: self._set_playback_state(not self.is_playing)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a: self.change_segment(-1)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: self.change_segment(1)
                
                elif self.state in ['CORRELATION_VIEW', 'GLOBAL_CORRELATION_VIEW']:
                    is_global = self.state == 'GLOBAL_CORRELATION_VIEW'
                    
                    if event.key in [pygame.K_w, pygame.K_s]:
                        correlation_data = self.global_correlation_data if is_global else self.correlation_data
                        selected_feature = self.selected_global_correlation_feature if is_global else self.selected_correlation_feature
                        if correlation_data and selected_feature:
                            sorted_features = sorted(correlation_data.items(), key=lambda item: item[1]['label'])
                            feature_keys = [item[0] for item in sorted_features]
                            if feature_keys:
                                try:
                                    current_index = feature_keys.index(selected_feature)
                                    direction = -1 if event.key == pygame.K_w else 1
                                    new_index = (current_index + direction) % len(feature_keys)
                                    new_selection = feature_keys[new_index]
                                    if is_global: self.selected_global_correlation_feature = new_selection
                                    else: self.selected_correlation_feature = new_selection
                                except ValueError:
                                    if is_global: self.selected_global_correlation_feature = feature_keys[0]
                                    else: self.selected_correlation_feature = feature_keys[0]
                    
                    if event.key in [pygame.K_a, pygame.K_d]:
                        selected_metric = self.selected_global_score_metric if is_global else self.selected_score_metric
                        metrics = ['f1', 'p', 'r']
                        try:
                            current_index = metrics.index(selected_metric)
                            direction = -1 if event.key == pygame.K_a else 1
                            new_index = (current_index + direction) % len(metrics)
                            new_selection = metrics[new_index]
                            if is_global: self.selected_global_score_metric = new_selection
                            else: self.selected_score_metric = new_selection
                        except ValueError:
                            if is_global: self.selected_global_score_metric = metrics[0]
                            else: self.selected_score_metric = metrics[0]

                    if event.key == pygame.K_n:
                        self.correlation_normalize = not self.correlation_normalize
                    elif event.key == pygame.K_r:
                         self.show_regression_line = not self.show_regression_line
                    elif event.key == pygame.K_1:
                        self.correlation_plot_type = 'scatter'
                    elif event.key == pygame.K_2:
                        self.correlation_plot_type = 'hist_feature'
                    elif event.key == pygame.K_3:
                        self.correlation_plot_type = 'hist_score'

                if event.key == pygame.K_ESCAPE: self._go_back()

    def _go_back(self):
        if self.plot_type_dropdown_open: self.plot_type_dropdown_open = False
        elif self.show_feature_popup: self.show_feature_popup = False
        elif self.state == 'GLOBAL_CORRELATION_VIEW':
            self.state = 'FILE_SELECT'; self.global_correlation_data = {}
        elif self.state == 'CORRELATION_VIEW':
            self.state = 'DETAIL_VIEW'; self.correlation_data = {}
        elif self.state == 'DETAIL_VIEW': self.state = 'FILE_SELECT'; self.detail_data = {}; self.segment_scroll_y = 0
        elif self.state == 'PIANO_ROLL_VIEW': self.state = 'DETAIL_VIEW'; self._set_playback_state(False); self.piano_roll_data = {}

    def update(self):
        if self.state == 'FILE_SELECT': max_scroll = max(0, len(self.audio_folders_data) * 40 - (SCREEN_HEIGHT - 100)); self.file_scroll_y = max(0, min(self.file_scroll_y, max_scroll))
        elif self.state == 'DETAIL_VIEW' and self.detail_data.get('segment_list'): max_scroll = max(0, len(self.detail_data['segment_list']) * 35 - (SCREEN_HEIGHT - 135)); self.segment_scroll_y = max(0, min(self.segment_scroll_y, max_scroll))
        if self.state == 'PIANO_ROLL_VIEW' and self.is_playing and not self.is_scrubbing:
            elapsed_ticks = pygame.time.get_ticks() - self.play_start_tick
            self.current_playback_ms = self.play_start_offset_ms + elapsed_ticks
            cut_length_ms = self.piano_roll_data.get('cut_length', 0) * 1000.0
            if cut_length_ms > 0 and self.current_playback_ms >= cut_length_ms: self.is_playing = False; pygame.mixer.music.stop(); self.current_playback_ms = self.play_start_offset_ms

    def _set_playback_state(self, should_play):
        if should_play:
            if not self.is_playing:
                full_audio_slice = self.piano_roll_data.get('full_audio_slice')
                if full_audio_slice:
                    start_ms = int(self.current_playback_ms); playback_slice = full_audio_slice[start_ms:]
                    sound_stream = io.BytesIO(); playback_slice.export(sound_stream, format="wav"); sound_stream.seek(0)
                    pygame.mixer.music.load(sound_stream); pygame.mixer.music.play()
                    self.play_start_tick = pygame.time.get_ticks(); self.play_start_offset_ms = self.current_playback_ms; self.is_playing = True
        else:
            if self.is_playing: self.is_playing = False; pygame.mixer.music.stop()

    def handle_table_header_click(self, pos, y_pos, columns, data_list, sort_config):
        x_offset = 50
        for _, col_info in columns.items():
            header_rect = pygame.Rect(x_offset, y_pos, col_info['width'], 40)
            if header_rect.collidepoint(pos):
                if sort_config['key'] == col_info['key']: sort_config['ascending'] = not sort_config['ascending']
                else: sort_config['key'] = col_info['key']; sort_config['ascending'] = True
                self._sort_list(data_list, sort_config); return True
            x_offset += col_info['width']
        return False

    def handle_file_select_clicks(self, pos):
        GLOBAL_CORR_BUTTON_RECT = pygame.Rect(SCREEN_WIDTH - 270, 10, 250, 40)
        if GLOBAL_CORR_BUTTON_RECT.collidepoint(pos):
            self.calculate_global_correlations()
            self.state = 'GLOBAL_CORRELATION_VIEW'
            return

        if self.handle_table_header_click(pos, 50, self.FILE_COLUMNS, self.audio_folders_data, self.file_list_sort_config): return
        list_rect = pygame.Rect(50, 90, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        if list_rect.collidepoint(pos):
            for i, folder_data in enumerate(self.audio_folders_data):
                item_rect = pygame.Rect(list_rect.x, list_rect.y + i * 40 - self.file_scroll_y, list_rect.width, 40)
                if item_rect.collidepoint(pos):
                    self.selected_file_info = folder_data; self.load_detail_data(); self._update_dynamic_segment_columns()
                    self.state = 'DETAIL_VIEW'; break

    def _handle_feature_popup_clicks(self, pos):
        if self.feature_popup_close_button_rect.collidepoint(pos):
            self.show_feature_popup = False; self._update_dynamic_segment_columns(); return
        for key, rect in self.feature_checkbox_rects.items():
            if rect.collidepoint(pos):
                if key in self.selected_features: self.selected_features.remove(key)
                else: self.selected_features.add(key)
                return

    def handle_detail_view_clicks(self, pos):
        if self.show_feature_popup: self._handle_feature_popup_clicks(pos); return
        if self.feature_select_button_rect.collidepoint(pos): self.show_feature_popup = True; return
        if pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40).collidepoint(pos): self._go_back(); return
        
        if self.detail_corr_button_rect.collidepoint(pos):
            self.calculate_correlations()
            self.state = 'CORRELATION_VIEW'
            return

        if self.handle_table_header_click(pos, 95, self.dynamic_segment_columns, self.detail_data.get('segment_list',[]), self.segment_list_sort_config): return
        list_rect = pygame.Rect(50, 135, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 135)
        if list_rect.collidepoint(pos) and self.detail_data.get('segment_list'):
            for i, seg_data in enumerate(self.detail_data['segment_list']):
                item_rect = pygame.Rect(list_rect.x, list_rect.y + i * 35 - self.segment_scroll_y, list_rect.width, 35)
                if item_rect.collidepoint(pos):
                    pygame.mixer.music.stop(); self.current_playback_ms = 0.0; self.is_playing = False; self.is_scrubbing = False
                    self.piano_roll_view_options['show_gt'] = True; self.piano_roll_view_options['show_pred'] = True
                    self.selected_segment = seg_data['num']; self.load_piano_roll_data(); self.state = 'PIANO_ROLL_VIEW'; break

    def handle_piano_roll_space_key(self): self._set_playback_state(not self.is_playing)

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
        if STOP_RECT.collidepoint(pos): self.current_playback_ms = 0.0; self._set_playback_state(False)
        if self.piano_roll_view_options['gt_checkbox_rect'].collidepoint(pos): self.piano_roll_view_options['show_gt'] = not self.piano_roll_view_options['show_gt']; return
        if self.piano_roll_view_options['pred_checkbox_rect'].collidepoint(pos): self.piano_roll_view_options['show_pred'] = not self.piano_roll_view_options['show_pred']; return
        
        roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250)
        if roll_rect.collidepoint(pos): self.is_scrubbing = True; self._set_playback_state(False); self.handle_piano_roll_scrub(pos)

    def handle_correlation_view_clicks(self, pos):
        is_global = self.state == 'GLOBAL_CORRELATION_VIEW'

        # --- DropDown Menu Click Handling (must be checked first) ---
        main_dropdown_rect = self.dropdown_rects.get('main')
        if main_dropdown_rect and main_dropdown_rect.collidepoint(pos):
            self.plot_type_dropdown_open = not self.plot_type_dropdown_open
            return

        if self.plot_type_dropdown_open:
            for p_type, rect in self.dropdown_rects.get('options', {}).items():
                if rect.collidepoint(pos):
                    self.correlation_plot_type = p_type
                    self.plot_type_dropdown_open = False
                    return
            # If the dropdown is open and the click is outside, close it and check other buttons
            self.plot_type_dropdown_open = False
        # -------------------------------------------------------------

        back_button_rect = pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40)
        if back_button_rect.collidepoint(pos):
            self._go_back(); return
        
        metric_buttons = self.global_correlation_metric_buttons if is_global else self.correlation_metric_buttons
        for metric, rect in metric_buttons.items():
            if rect.collidepoint(pos):
                if is_global: self.selected_global_score_metric = metric
                else: self.selected_score_metric = metric
                return

        feature_buttons = self.global_correlation_feature_buttons if is_global else self.correlation_feature_buttons
        for feature_key, rect in feature_buttons.items():
            if rect.collidepoint(pos):
                if is_global: self.selected_global_correlation_feature = feature_key
                else: self.selected_correlation_feature = feature_key
                return

        for control_key, rect in self.correlation_controls.items():
            if rect.collidepoint(pos):
                if control_key == 'normalize':
                    self.correlation_normalize = not self.correlation_normalize
                elif control_key == 'regression':
                    self.show_regression_line = not self.show_regression_line
                return

    def handle_piano_roll_scrub(self, pos):
        roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250)
        if self.piano_roll_data.get('cut_length'):
            time_ratio = (pos[0] - roll_rect.x) / roll_rect.width
            seek_time_sec = time_ratio * self.piano_roll_data.get('cut_length', 0)
            max_time_ms = self.piano_roll_data['cut_length'] * 1000.0
            self.current_playback_ms = max(0, min(seek_time_sec * 1000.0, max_time_ms))

    def change_segment(self, direction):
        segment_list = self.detail_data.get('segment_list');
        if not segment_list: return
        try: current_index = [i for i, seg in enumerate(segment_list) if seg['num'] == self.selected_segment][0]
        except IndexError: print(f"Error: Current segment {self.selected_segment} not found."); return
        new_index = current_index + direction
        if 0 <= new_index < len(segment_list):
            self.selected_segment = segment_list[new_index]['num']
            self._set_playback_state(False); self.current_playback_ms = 0.0; self.is_scrubbing = False
            self.load_piano_roll_data()
        else: print(f"No more segments in direction {direction}.")

    def load_detail_data(self):
        if not self.global_metadata: print("Error: Global metadata not loaded."); self.detail_data = {}; return
        try:
            selected_filename = self.selected_file_info['name']; print(f"Loading detail data for {selected_filename}...")
            total_segment_count = self.global_metadata.get('items', {}).get(selected_filename, {}).get('num_midi_segments_created', 0)
            file_metadata = self._load_file_evaluation_metadata(selected_filename); eval_data_for_file = file_metadata.get('segments', {})
            segment_list = []
            for seg_num in range(1, total_segment_count + 1):
                scores = eval_data_for_file.get(str(seg_num), {})
                segment_list.append({'num': seg_num, 'p': scores.get('Precision', 0.0), 'r': scores.get('Recall', 0.0), 'f1': scores.get('F1-Score', 0.0)})
            print("Calculating ground truth features for all segments...")
            base_name = selected_filename.replace('_16020Hz', ''); gt_path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'midi', f"{base_name}.mid")
            gt_midi = None
            if os.path.exists(gt_path):
                try: gt_midi = pretty_midi.PrettyMIDI(gt_path)
                except Exception as e: print(f"Could not load ground truth MIDI {gt_path}: {e}")
            for seg_data in segment_list:
                seg_data['features'] = {};
                if not gt_midi: continue
                seg_num = seg_data['num']; cut_length = self.global_metadata['midi_extractor_settings']['segment_length_sec']
                start_time = (seg_num - 1) * cut_length; end_time = start_time + cut_length
                gt_notes_in_segment = []
                for inst in gt_midi.instruments:
                    for note in inst.notes:
                        if note.start < end_time and note.end > start_time:
                            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=max(0, note.start - start_time), end=min(cut_length, note.end - start_time))
                            if new_note.end > new_note.start: gt_notes_in_segment.append(new_note)
                if gt_notes_in_segment:
                    temp_midi = pretty_midi.PrettyMIDI(); instrument = pretty_midi.Instrument(program=0); instrument.notes.extend(gt_notes_in_segment)
                    temp_midi.instruments.append(instrument); temp_midi_path = "temp_gt_segment_for_features.mid"
                    try:
                        temp_midi.write(temp_midi_path); features_obj = MidiFeatures(temp_midi_path)
                        if features_obj.available: seg_data['features'] = features_obj.numeric_features
                    except Exception as e: print(f"Error calculating features for segment {seg_num}: {e}")
                    finally:
                        if os.path.exists(temp_midi_path): os.remove(temp_midi_path)
            print("Feature calculation complete."); self.detail_data['segment_list'] = segment_list
            self._sort_list(self.detail_data['segment_list'], self.segment_list_sort_config)
        except Exception as e: print(f"Error loading detail data: {e}"); self.detail_data = {}

    def calculate_correlations(self):
        print("Calculating feature-score correlations for segments...")
        self.correlation_data = {}
        segment_list = self.detail_data.get('segment_list', [])
        if not segment_list:
            print("No segment data available for correlation."); return

        all_p = [s['p'] for s in segment_list if s.get('p') is not None]
        all_r = [s['r'] for s in segment_list if s.get('r') is not None]
        all_f1 = [s['f1'] for s in segment_list if s.get('f1') is not None]

        min_p = min(all_p) if all_p else 1.0
        min_r = min(all_r) if all_r else 1.0
        min_f1 = min(all_f1) if all_f1 else 1.0

        denom_p = 1 - min_p
        denom_r = 1 - min_r
        denom_f1 = 1 - min_f1

        for feature_label, (feature_key, _, _) in self.features_info.items():
            data_points = []
            for segment in segment_list:
                feature_value = segment.get('features', {}).get(feature_key)
                p, r, f1 = segment.get('p'), segment.get('r'), segment.get('f1')
                if feature_value is not None and all(s is not None for s in [p, r, f1]):
                    tp = ((1 - p) / denom_p * SCORE_SCALE_CONSTANT) if denom_p > 0 else 0
                    tr = ((1 - r) / denom_r * SCORE_SCALE_CONSTANT) if denom_r > 0 else 0
                    tf1 = ((1 - f1) / denom_f1 * SCORE_SCALE_CONSTANT) if denom_f1 > 0 else 0
                    data_points.append({
                        'feature': feature_value,
                        'p': tp, 'r': tr, 'f1': tf1,
                        'original_p': p, 'original_r': r, 'original_f1': f1,
                        'segment': segment['num']
                    })

            if len(data_points) < 2: continue

            correlations = {}
            feature_values = np.array([d['feature'] for d in data_points])
            original_scores = {m: np.array([d[f'original_{m}'] for d in data_points]) for m in ['p', 'r', 'f1']}
            
            for metric in ['p', 'r', 'f1']:
                score_values = original_scores[metric]
                if np.std(feature_values) == 0 or np.std(score_values) == 0:
                    corr_coeff = 0.0
                else:
                    corr_matrix = np.corrcoef(feature_values, score_values)
                    corr_coeff = corr_matrix[0, 1]
                correlations[metric] = corr_coeff if not np.isnan(corr_coeff) else 0.0

            self.correlation_data[feature_key] = {'label': feature_label, 'correlations': correlations, 'points': data_points}
        
        if not self.selected_correlation_feature and self.correlation_data:
            self.selected_correlation_feature = next(iter(self.correlation_data.keys()))
        print("Segment correlation calculation complete.")

    def calculate_global_correlations(self):
        print("Calculating global feature-score correlations...")
        self.global_correlation_data = {}
        file_list = self.audio_folders_data
        features_db = self.groundtruth_features_metadata
        if not file_list or not features_db:
            print("No file or feature data available for global correlation.")
            return

        all_p = [f['p'] for f in file_list if f.get('p') is not None]
        all_r = [f['r'] for f in file_list if f.get('r') is not None]
        all_f1 = [f['f1'] for f in file_list if f.get('f1') is not None]

        min_p = min(all_p) if all_p else 1.0
        min_r = min(all_r) if all_r else 1.0
        min_f1 = min(all_f1) if all_f1 else 1.0

        denom_p = 1 - min_p
        denom_r = 1 - min_r
        denom_f1 = 1 - min_f1

        for feature_label, (feature_key, _, _) in self.features_info.items():
            data_points = []
            for file_info in file_list:
                record_time = file_info['name'].replace('_16020Hz', '')
                feature_file_key = f"{record_time}.mid"
                feature_data = features_db.get(feature_file_key, {})
                if not feature_data.get('available'): continue
                feature_value = feature_data.get('numeric_features', {}).get(feature_key)
                p, r, f1 = file_info.get('p'), file_info.get('r'), file_info.get('f1')

                if feature_value is not None and all(s is not None for s in [p, r, f1]):
                    tp = ((1 - p) / denom_p * SCORE_SCALE_CONSTANT) if denom_p > 0 else 0
                    tr = ((1 - r) / denom_r * SCORE_SCALE_CONSTANT) if denom_r > 0 else 0
                    tf1 = ((1 - f1) / denom_f1 * SCORE_SCALE_CONSTANT) if denom_f1 > 0 else 0
                    data_points.append({
                        'feature': feature_value, 
                        'p': tp, 'r': tr, 'f1': tf1,
                        'original_p': p, 'original_r': r, 'original_f1': f1,
                        'piece': file_info.get('piece', 'N/A')
                    })

            if len(data_points) < 2: continue
            
            correlations = {}
            feature_values = np.array([d['feature'] for d in data_points])
            original_scores = {m: np.array([d[f'original_{m}'] for d in data_points]) for m in ['p', 'r', 'f1']}

            for metric in ['p', 'r', 'f1']:
                score_values = original_scores[metric]
                if np.std(feature_values) == 0 or np.std(score_values) == 0:
                    corr_coeff = 0.0
                else:
                    corr_matrix = np.corrcoef(feature_values, score_values)
                    corr_coeff = corr_matrix[0, 1]
                correlations[metric] = corr_coeff if not np.isnan(corr_coeff) else 0.0

            self.global_correlation_data[feature_key] = {
                'label': feature_label,
                'correlations': correlations,
                'points': data_points
            }
        
        if not self.selected_global_correlation_feature and self.global_correlation_data:
            self.selected_global_correlation_feature = next(iter(self.global_correlation_data.keys()))
        print("Global correlation calculation complete.")

    def load_piano_roll_data(self):
        if not self.global_metadata: print("Warning: Global metadata not loaded."); self.piano_roll_data = {}; return
        pred_notes, gt_notes_in_segment = [], []; matched_ref_indices, matched_est_indices = set(), set()
        frame_scores = {'p': 0.0, 'r': 0.0, 'f1': 0.0}; frame_counts = {'TP': 0, 'FP': 0, 'TN': 0}
        audio_slice, cut_length, gt_features, pred_features = None, 0, None, None
        try:
            cut_length = self.global_metadata['midi_extractor_settings']['segment_length_sec']; start_time = (self.selected_segment - 1) * cut_length; end_time = start_time + cut_length
            selected_filename = self.selected_file_info['name']; base_name = selected_filename.replace('_16020Hz', '')
            pred_path = os.path.join(PREDICTED_MIDI_PATH, selected_filename, f"{self.selected_segment}.mid"); gt_path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'midi', f"{base_name}.mid")
            try:
                if os.path.exists(pred_path):
                    pred_midi = pretty_midi.PrettyMIDI(pred_path)
                    if pred_midi.instruments and pred_midi.instruments[0].notes: pred_notes = pred_midi.instruments[0].notes
                else: print(f"Warning: Predicted MIDI not found at '{pred_path}'")
            except Exception as e: print(f"Error loading predicted MIDI: {e}")
            try:
                if os.path.exists(gt_path):
                    gt_midi = pretty_midi.PrettyMIDI(gt_path)
                    for inst in gt_midi.instruments:
                        for note in inst.notes:
                            if note.start < end_time and note.end > start_time:
                                new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=max(0, note.start - start_time), end=min(cut_length, note.end - start_time))
                                if new_note.end > new_note.start: gt_notes_in_segment.append(new_note)
                else: print(f"Warning: Ground truth MIDI not found at '{gt_path}'")
            except Exception as e: print(f"Error loading ground truth MIDI: {e}")
            if gt_notes_in_segment:
                temp_gt_midi = pretty_midi.PrettyMIDI(); instrument = pretty_midi.Instrument(program=0); instrument.notes.extend(gt_notes_in_segment)
                temp_gt_midi.instruments.append(instrument); temp_gt_path = "temp_gt_segment_features.mid"; temp_gt_midi.write(temp_gt_path)
                gt_features_obj = MidiFeatures(temp_gt_path)
                if gt_features_obj.available: gt_features = gt_features_obj.numeric_features
                os.remove(temp_gt_path)
            if os.path.exists(pred_path) and pred_notes:
                pred_features_obj = MidiFeatures(pred_path)
                if pred_features_obj.available: pred_features = pred_features_obj.numeric_features
            if gt_notes_in_segment and pred_notes:
                ref_intervals_note = np.array([[n.start, n.end] for n in gt_notes_in_segment]); ref_pitches_note = np.array([n.pitch for n in gt_notes_in_segment])
                est_intervals_note = np.array([[n.start, n.end] for n in pred_notes]); est_pitches_note = np.array([n.pitch for n in pred_notes])
                if ref_intervals_note.size > 0 and est_intervals_note.size > 0:
                    matching = mir_eval.transcription.match_notes(ref_intervals_note, ref_pitches_note, est_intervals_note, est_pitches_note)
                    matched_ref_indices = {m[0] for m in matching}; matched_est_indices = {m[1] for m in matching}
            frame_len_sec = 0.01; frame_times = np.arange(0, cut_length, frame_len_sec); total_tp, total_fp, total_tn = 0, 0, 0
            for time in frame_times:
                true_pitches = {note.pitch for note in gt_notes_in_segment if note.start <= time < note.end}; pred_pitches = {note.pitch for note in pred_notes if note.start <= time < note.end}
                total_tp += len(true_pitches.intersection(pred_pitches)); total_fp += len(pred_pitches.difference(true_pitches))
                total_tn += len(true_pitches.difference(pred_pitches))
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r = total_tp / (total_tp + total_tn) if (total_tp + total_tn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0; frame_scores = {'p': p, 'r': r, 'f1': f1}
            frame_counts = {'TP': total_tp, 'FP': total_fp, 'TN': total_tn}
            audio_path = os.path.join(GROUNDTRUTH_MIDI_PATH, 'audio', f"{selected_filename}.wav")
            if os.path.exists(audio_path): full_audio = AudioSegment.from_wav(audio_path); audio_slice = full_audio[start_time * 1000 : end_time * 1000]
            else: print(f"Warning: Audio file not found at '{audio_path}'")
        except Exception as e: print(f"An unexpected error occurred in load_piano_roll_data: {e}")
        finally:
            self.piano_roll_data = {'pred_notes': pred_notes, 'gt_notes': gt_notes_in_segment, 'matched_ref_indices': matched_ref_indices, 'matched_est_indices': matched_est_indices, 'cut_length': cut_length, 'frame_scores': frame_scores, 'frame_counts': frame_counts, 'full_audio_slice': audio_slice, 'gt_features': gt_features, 'pred_features': pred_features}

    def draw(self):
        screen.fill(COLORS['background'])
        if self.state == 'FILE_SELECT': self.draw_file_select_screen()
        elif self.state == 'DETAIL_VIEW': self.draw_detail_view_screen()
        elif self.state == 'PIANO_ROLL_VIEW': self.draw_piano_roll_screen()
        elif self.state == 'CORRELATION_VIEW':
            self._draw_unified_correlation_view(is_global=False)
        elif self.state == 'GLOBAL_CORRELATION_VIEW':
            self._draw_unified_correlation_view(is_global=True)

        # Dropdown menu must be drawn last to appear on top of other elements
        if self.state in ['CORRELATION_VIEW', 'GLOBAL_CORRELATION_VIEW']:
            self._draw_plot_type_dropdown()

        self._draw_tooltip(); pygame.display.flip()

    def _update_dynamic_segment_columns(self):
        self.dynamic_segment_columns = self.SEGMENT_COLUMNS_BASE.copy()
        ordered_features = sorted(list(self.features_info.items()), key=lambda item: item[0])
        for label, (key, _, _) in ordered_features:
            if key in self.selected_features: self.dynamic_segment_columns[key] = {'label': label, 'width': 150, 'key': key}
        total_width = sum(c['width'] for c in self.dynamic_segment_columns.values())
        if total_width > SCREEN_WIDTH - 100: print(f"Warning: Total column width ({total_width}) exceeds screen space.")

    def _draw_table_header(self, y_pos, columns, sort_config):
        header_rect = pygame.Rect(50, y_pos, SCREEN_WIDTH - 100, 40); pygame.draw.rect(screen, COLORS['header'], header_rect)
        x_offset = 50
        for _, col_info in columns.items():
            label = col_info['label']
            if col_info['key'] == sort_config['key']: label += " +" if sort_config['ascending'] else " -"
            text_surf = FONT_SMALL.render(label, True, COLORS['text'])
            text_rect = text_surf.get_rect(centerx = x_offset + col_info['width'] / 2, centery = y_pos + 20)
            screen.blit(text_surf, text_rect); x_offset += col_info['width']

    def draw_file_select_screen(self):
        screen.blit(FONT_MAIN.render("Select an Audio File", True, COLORS['text']), (50, 10))
        
        mouse_pos = pygame.mouse.get_pos()
        GLOBAL_CORR_BUTTON_RECT = pygame.Rect(SCREEN_WIDTH - 270, 10, 250, 35)
        btn_color = COLORS['button_hover'] if GLOBAL_CORR_BUTTON_RECT.collidepoint(mouse_pos) else COLORS['button']
        pygame.draw.rect(screen, btn_color, GLOBAL_CORR_BUTTON_RECT, border_radius=5)
        btn_text = FONT_SMALL.render("Global Correlation Analysis", True, COLORS['text'])
        btn_text_rect = btn_text.get_rect(center=GLOBAL_CORR_BUTTON_RECT.center)
        screen.blit(btn_text, btn_text_rect)

        self._draw_table_header(50, self.FILE_COLUMNS, self.file_list_sort_config)
        list_rect = pygame.Rect(50, 90, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        for i, folder_data in enumerate(self.audio_folders_data):
            y_pos = list_rect.y + i * 40 - self.file_scroll_y
            if list_rect.top <= y_pos < list_rect.bottom:
                item_rect = pygame.Rect(list_rect.x, y_pos, list_rect.width, 40)
                if item_rect.collidepoint(pygame.mouse.get_pos()): pygame.draw.rect(screen, COLORS['button_hover'], item_rect)
                x_offset = 50; record_time_full = folder_data['name'].replace('_16020Hz', '')
                try: segs = record_time_full.split('_'); display_name = segs[0][2:] + ' ' + segs[1].replace('-',':')
                except IndexError: display_name = record_time_full
                col_data = {'index': str(i + 1), 'name': display_name, 'p': f"{folder_data['p']:.4f}", 'r': f"{folder_data['r']:.4f}", 'f1': f"{folder_data['f1']:.4f}", 'piece': folder_data.get('piece', 'N/A'), 'player': folder_data.get('player', 'N/A'), 'skill': folder_data.get('skill', 'N/A'), 'split': folder_data.get('split', 'N/A')}
                for _, col_info in self.FILE_COLUMNS.items():
                    text_surf = FONT_SMALL.render(str(col_data.get(col_info['key'], '')), True, COLORS['text'])
                    screen.blit(text_surf, (x_offset + 5, item_rect.y + 10)); x_offset += col_info['width']

    def draw_detail_view_screen(self):
        back_button_rect = pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40)
        fs_button_rect = self.feature_select_button_rect
        corr_button_rect = self.detail_corr_button_rect
        mouse_pos = pygame.mouse.get_pos()
        
        for rect, text in [(back_button_rect, "<< Back"), (fs_button_rect, "Select Columns"), (corr_button_rect, "Correlations")]:
            color = COLORS['button_hover'] if rect.collidepoint(mouse_pos) else COLORS['button']
            pygame.draw.rect(screen, color, rect, border_radius=5)
            text_surf = FONT_SMALL.render(text, True, COLORS['text'])
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)
        
        info = self.selected_file_info
        screen.blit(FONT_MAIN.render(f"Piece: {info.get('piece', 'N/A')}", True, COLORS['text']), (50, 10))
        screen.blit(FONT_SMALL.render(f"Player: {info.get('player', 'N/A')} ({info.get('skill', 'N/A')})   |   Split: {info.get('split', 'N/A')}", True, COLORS['text']), (50, 45))
        screen.blit(FONT_TINY.render(f"File: {info.get('name', '')}", True, (180, 180, 180)), (50, 70))
        self._draw_table_header(95, self.dynamic_segment_columns, self.segment_list_sort_config)
        list_rect = pygame.Rect(50, 135, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 135)
        if self.detail_data.get('segment_list'):
            for i, seg_data in enumerate(self.detail_data['segment_list']):
                y_pos = list_rect.y + i * 35 - self.segment_scroll_y
                if list_rect.top <= y_pos < list_rect.bottom:
                    item_rect = pygame.Rect(list_rect.x, y_pos, sum(c['width'] for c in self.dynamic_segment_columns.values()), 35)
                    if item_rect.collidepoint(pygame.mouse.get_pos()): pygame.draw.rect(screen, COLORS['button_hover'], item_rect)
                    x_offset = 50; base_data = {'index': str(i + 1), 'num': str(seg_data['num']), 'p': f"{seg_data['p']:.4f}", 'r': f"{seg_data['r']:.4f}", 'f1': f"{seg_data['f1']:.4f}"}
                    for _, col_info in self.dynamic_segment_columns.items():
                        key = col_info['key']; val_str = "";
                        if key in base_data: val_str = base_data.get(key, '')
                        else:
                            feature_val = seg_data.get('features', {}).get(key)
                            fmt = '{}'
                            for f_label, f_info_tuple in self.features_info.items():
                                if f_info_tuple[0] == key:
                                    fmt = f_info_tuple[1]
                                    break
                            
                            if feature_val is not None:
                                try: val_str = fmt.format(feature_val)
                                except (ValueError, TypeError): val_str = str(round(feature_val, 2)) if isinstance(feature_val, float) else str(feature_val)
                            else: val_str = "N/A"
                        text_surf = FONT_SMALL.render(val_str, True, COLORS['text']); text_rect = text_surf.get_rect(centerx = x_offset + col_info['width'] / 2, centery = item_rect.centery)
                        screen.blit(text_surf, text_rect); x_offset += col_info['width']
        if self.show_feature_popup: self._draw_feature_popup()

    def _draw_feature_popup(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill(COLORS['popup_overlay']); screen.blit(overlay, (0, 0))
        popup_rect = self.feature_popup_rect; pygame.draw.rect(screen, COLORS['header'], popup_rect, border_radius=10); pygame.draw.rect(screen, COLORS['grid_line'], popup_rect, 2, border_radius=10)
        title_surf = FONT_MAIN.render("Select Feature Columns", True, COLORS['text']); title_rect = title_surf.get_rect(centerx=popup_rect.centerx, y=popup_rect.y + 20); screen.blit(title_surf, title_rect)
        self.feature_checkbox_rects.clear(); start_y = popup_rect.y + 80; x_col1 = popup_rect.x + 30; x_col2 = popup_rect.centerx + 30; checkbox_size = 20; row_height = 35
        features = list(self.features_info.items()); mid_point = (len(features) + 1) // 2
        for i, (label, (key, _, _)) in enumerate(features):
            col_x = x_col1 if i < mid_point else x_col2; row_num = i if i < mid_point else i - mid_point; y_pos = start_y + row_num * row_height
            cb_rect = pygame.Rect(col_x, y_pos, checkbox_size, checkbox_size); self.feature_checkbox_rects[key] = cb_rect
            pygame.draw.rect(screen, COLORS['checkbox_border'], cb_rect, 2, border_radius=3)
            if key in self.selected_features: pygame.draw.rect(screen, COLORS['checkbox_checked'], cb_rect.inflate(-8, -8), border_radius=3)
            screen.blit(FONT_SMALL.render(label, True, COLORS['text']), (cb_rect.x + 30, cb_rect.y - 2))
        close_rect = self.feature_popup_close_button_rect; mouse_pos = pygame.mouse.get_pos()
        if close_rect.collidepoint(mouse_pos): pygame.draw.rect(screen, COLORS['button_hover'], close_rect, border_radius=5)
        else: pygame.draw.rect(screen, COLORS['button'], close_rect, border_radius=5)
        close_surf = FONT_SMALL.render("Close", True, COLORS['text']); close_text_rect = close_surf.get_rect(center=close_rect.center); screen.blit(close_surf, close_text_rect)

    def check_feature_hover(self, pos):
        table_rect = pygame.Rect(250, SCREEN_HEIGHT - 150 + 20, SCREEN_WIDTH - 300, 90)
        num_cols = len(self.features_info); 
        if num_cols == 0: return
        col_width = table_rect.width / num_cols; x_offset = table_rect.x
        is_hovering_feature = False
        for label, (key, fmt, desc) in self.features_info.items():
            cell_rect = pygame.Rect(x_offset, table_rect.y, col_width, table_rect.height)
            if cell_rect.collidepoint(pos):
                self.active_tooltip = {'text': desc, 'pos': pos}
                is_hovering_feature = True
                break
            x_offset += col_width
        if not is_hovering_feature:
            if self.active_tooltip and self.active_tooltip.get('text') in [d[2] for d in self.features_info.values()]:
                self.active_tooltip = None

    def _draw_tooltip(self):
        if not self.active_tooltip: return
        text_surf = FONT_TINY.render(self.active_tooltip['text'], True, COLORS['tooltip_text']); padding = 8
        bg_rect = text_surf.get_rect().inflate(padding * 2, padding * 2); mouse_pos = self.active_tooltip['pos']
        bg_rect.bottomleft = (mouse_pos[0] + 15, mouse_pos[1] - 5)
        if bg_rect.right > SCREEN_WIDTH: bg_rect.right = SCREEN_WIDTH - 5
        if bg_rect.left < 0: bg_rect.left = 5;
        if bg_rect.top < 0: bg_rect.top = 5
        pygame.draw.rect(screen, COLORS['tooltip_bg'], bg_rect, border_radius=5); screen.blit(text_surf, (bg_rect.x + padding, bg_rect.y + padding))

    def _draw_legend_and_controls(self, start_x, start_y):
        y_offset = 0; show_gt = self.piano_roll_view_options['show_gt']; show_pred = self.piano_roll_view_options['show_pred']; checkbox_size = 20
        gt_rect = pygame.Rect(start_x, start_y + y_offset, checkbox_size, checkbox_size); self.piano_roll_view_options['gt_checkbox_rect'] = gt_rect
        pygame.draw.rect(screen, COLORS['checkbox_border'], gt_rect, 2, border_radius=3); screen.blit(FONT_SMALL.render("GT MIDI (W)", True, COLORS['text']), (gt_rect.x + 30, gt_rect.y - 2))
        if show_gt: pygame.draw.rect(screen, COLORS['checkbox_checked'], gt_rect.inflate(-6, -6), border_radius=3)
        y_offset += 35; pred_rect = pygame.Rect(start_x, start_y + y_offset, checkbox_size, checkbox_size); self.piano_roll_view_options['pred_checkbox_rect'] = pred_rect
        pygame.draw.rect(screen, COLORS['checkbox_border'], pred_rect, 2, border_radius=3); screen.blit(FONT_SMALL.render("Pred MIDI (S)", True, COLORS['text']), (pred_rect.x + 30, pred_rect.y - 2))
        if show_pred: pygame.draw.rect(screen, COLORS['checkbox_checked'], pred_rect.inflate(-6, -6), border_radius=3)
        y_offset += 55; legend_items = []
        if show_gt and show_pred: legend_items = [("Correct (TP)", COLORS['tp_green'], 'fill'), ("Extra Note (FP)", COLORS['fp_red_outline'], 'outline'), ("Missed Note (TN)", COLORS['fn_yellow_fill'], 'fill')]
        elif show_gt: legend_items = [("Ground Truth Note", COLORS['gt_only_color'], 'fill')]
        elif show_pred: legend_items = [("Predicted Note", COLORS['pred_only_color'], 'fill')]
        for label, color, style in legend_items:
            swatch_rect = pygame.Rect(start_x, start_y + y_offset, 20, 15)
            if style == 'fill': pygame.draw.rect(screen, color, swatch_rect, border_radius=3)
            else: pygame.draw.rect(screen, color, swatch_rect, 2, border_radius=3)
            screen.blit(FONT_SMALL.render(label, True, COLORS['text']), (start_x + 30, start_y + y_offset - 2)); y_offset += 30

    def _draw_midi_features_table(self, rect):
        gt_f = self.piano_roll_data.get('gt_features')
        if not gt_f: return
        pygame.draw.rect(screen, COLORS['header'], rect, border_radius=5); pygame.draw.rect(screen, COLORS['grid_line'], rect, 1, border_radius=5)
        screen.blit(FONT_SMALL.render("Ground Truth MIDI Features", True, COLORS['text']), (rect.x + 10, rect.y + 5))
        num_cols = len(self.features_info)
        if num_cols == 0: return
        col_width = rect.width / num_cols; table_y = rect.y + 35; cell_height = rect.height - 38; x_offset = rect.x
        for label, (key, fmt, desc) in self.features_info.items():
            cell_rect = pygame.Rect(x_offset, table_y, col_width, cell_height)
            label_surf = FONT_TINY.render(label, True, COLORS['gt_only_color']); label_rect = label_surf.get_rect(centerx=cell_rect.centerx, top=cell_rect.top + 8); screen.blit(label_surf, label_rect)
            val_str = "N/A"
            if key in gt_f:
                val = gt_f[key]
                try: val_str = fmt.format(val)
                except (ValueError, TypeError): val_str = str(val)
            value_surf = FONT_SMALL.render(val_str, True, COLORS['text']); value_rect = value_surf.get_rect(centerx=cell_rect.centerx, bottom=cell_rect.bottom - 8); screen.blit(value_surf, value_rect)
            x_offset += col_width
            if x_offset < rect.right: pygame.draw.line(screen, COLORS['grid_line'], (x_offset, table_y), (x_offset, table_y + cell_height))

    def _draw_piano_roll_stats(self):
        if not self.piano_roll_data: return
        scores = self.piano_roll_data.get('frame_scores', {}); counts = self.piano_roll_data.get('frame_counts', {})
        score_text = f"Frame Precision: {scores.get('p', 0):.4f}   Recall: {scores.get('r', 0):.4f}   F1-Score: {scores.get('f1', 0):.4f}"
        count_text = f"Frame TP: {counts.get('TP', 0)}   Extra Notes (FP): {counts.get('FP', 0)}   Missed Notes (TN): {counts.get('TN', 0)}"
        screen.blit(FONT_SMALL.render(score_text, True, COLORS['text']), (50, 45)); screen.blit(FONT_SMALL.render(count_text, True, COLORS['text']), (50, 70))

    def draw_piano_roll_screen(self):
        mouse_pos = pygame.mouse.get_pos()
        BACK_RECT = pygame.Rect(SCREEN_WIDTH - 130 - 10, 20, 120, 40)
        STOP_RECT = pygame.Rect(BACK_RECT.x - 120 - 10, 20, 120, 40)
        PLAY_RECT = pygame.Rect(STOP_RECT.x - 120 - 10, 20, 120, 40)
        NEXT_RECT = pygame.Rect(PLAY_RECT.x - 120 - 10, 20, 120, 40)
        PREV_RECT = pygame.Rect(NEXT_RECT.x - 120 - 10, 20, 120, 40)
        def draw_button(rect, text):
            color = COLORS['button_hover'] if rect.collidepoint(mouse_pos) else COLORS['button']; pygame.draw.rect(screen, color, rect, border_radius=5)
            text_surf = FONT_SMALL.render(text, True, COLORS['text']); text_rect = text_surf.get_rect(center=rect.center); screen.blit(text_surf, text_rect)
        
        draw_button(PREV_RECT, "<< Prev"); draw_button(NEXT_RECT, "Next >>"); draw_button(PLAY_RECT, "Pause" if self.is_playing else "Play"); draw_button(STOP_RECT, "Stop"); draw_button(BACK_RECT, "<< Back")
        info = self.selected_file_info; screen.blit(FONT_MAIN.render(f"Piece: {info.get('piece', 'N/A')} - Segment {self.selected_segment}", True, COLORS['text']), (50, 15))
        self._draw_piano_roll_stats(); roll_rect = pygame.Rect(250, 100, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 250); pygame.draw.rect(screen, COLORS['piano_roll_bg'], roll_rect)
        if not self.piano_roll_data: return
        min_pitch, max_pitch = 21, 108; pitch_span = max_pitch - min_pitch
        for pitch in range(min_pitch, max_pitch + 1):
            y = roll_rect.bottom - ((pitch - min_pitch) / pitch_span) * roll_rect.height; color = COLORS['c_note_line'] if self._get_note_name(pitch).startswith('C') else COLORS['grid_line']
            pygame.draw.line(screen, color, (roll_rect.left, y), (roll_rect.right, y), 1)
        for i in range(1, 9):
            pitch = 12 * (i + 1)
            if min_pitch <= pitch <= max_pitch: y = roll_rect.bottom - ((pitch - min_pitch) / pitch_span) * roll_rect.height; screen.blit(FONT_SMALL.render(f"C{i}", True, COLORS['text']), (roll_rect.left - 40, y - 10))
        note_height = roll_rect.height / pitch_span if pitch_span > 0 else 0; time_span = self.piano_roll_data.get('cut_length', 0); show_gt = self.piano_roll_view_options['show_gt']; show_pred = self.piano_roll_view_options['show_pred']
        if time_span > 0:
            if show_gt and show_pred:
                for gt_idx, gt_note in enumerate(self.piano_roll_data['gt_notes']):
                    x = roll_rect.x + (gt_note.start / time_span) * roll_rect.width; y = roll_rect.bottom - ((gt_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height; width = max(1, (gt_note.end - gt_note.start) / time_span * roll_rect.width)
                    color = COLORS['tp_green'] if gt_idx in self.piano_roll_data['matched_ref_indices'] else COLORS['fn_yellow_fill']; pygame.draw.rect(screen, color, (x, y, width, note_height), border_radius=2)
                for pred_idx, pred_note in enumerate(self.piano_roll_data['pred_notes']):
                    if pred_idx not in self.piano_roll_data['matched_est_indices']:
                        x = roll_rect.x + (pred_note.start / time_span) * roll_rect.width; y = roll_rect.bottom - ((pred_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height; width = max(1, (pred_note.end - pred_note.start) / time_span * roll_rect.width)
                        pygame.draw.rect(screen, COLORS['fp_red_outline'], (x, y, width, note_height), 2, border_radius=2)
            elif show_gt:
                for gt_note in self.piano_roll_data['gt_notes']: x, y = roll_rect.x + (gt_note.start / time_span) * roll_rect.width, roll_rect.bottom - ((gt_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height; width = max(1, (gt_note.end - gt_note.start) / time_span * roll_rect.width); pygame.draw.rect(screen, COLORS['gt_only_color'], (x, y, width, note_height), border_radius=2)
            elif show_pred:
                for pred_note in self.piano_roll_data['pred_notes']: x, y = roll_rect.x + (pred_note.start / time_span) * roll_rect.width, roll_rect.bottom - ((pred_note.pitch - min_pitch + 1) / pitch_span) * roll_rect.height; width = max(1, (pred_note.end - pred_note.start) / time_span * roll_rect.width); pygame.draw.rect(screen, COLORS['pred_only_color'], (x, y, width, note_height), border_radius=2)
        if time_span > 0:
            line_x = roll_rect.x + (self.current_playback_ms / (time_span * 1000.0)) * roll_rect.width
            if roll_rect.left <= line_x <= roll_rect.right: pygame.draw.line(screen, COLORS['playback_line'], (line_x, roll_rect.top), (line_x, roll_rect.bottom), 2)
        self._draw_legend_and_controls(30, 150); features_rect = pygame.Rect(250, roll_rect.bottom + 20, roll_rect.width, 90); self._draw_midi_features_table(features_rect)

    def _draw_plot_type_dropdown(self):
        mouse_pos = pygame.mouse.get_pos()
        main_rect = self.dropdown_rects.get('main')
        if not main_rect:
            return

        # Draw the main box
        color = COLORS['button_hover'] if main_rect.collidepoint(mouse_pos) or self.plot_type_dropdown_open else COLORS['button']
        pygame.draw.rect(screen, color, main_rect, border_radius=5)

        # Draw the currently selected text and an arrow
        selected_text = self.plot_types.get(self.correlation_plot_type, "Select Mode")
        text_surf = FONT_SMALL.render(selected_text, True, COLORS['text'])
        screen.blit(text_surf, (main_rect.x + 10, main_rect.centery - text_surf.get_height() / 2))
        arrow = "▲" if self.plot_type_dropdown_open else "▼"
        arrow_surf = FONT_SMALL.render(arrow, True, COLORS['text'])
        arrow_rect = arrow_surf.get_rect(centery=main_rect.centery, right=main_rect.right - 10)
        screen.blit(arrow_surf, arrow_rect)

        # If open, draw the options
        if self.plot_type_dropdown_open:
            self.dropdown_rects['options'] = {}
            option_y = main_rect.bottom
            for p_type, p_label in self.plot_types.items():
                option_rect = pygame.Rect(main_rect.x, option_y, main_rect.width, main_rect.height)
                self.dropdown_rects['options'][p_type] = option_rect
                
                opt_color = COLORS['button_hover'] if option_rect.collidepoint(mouse_pos) else COLORS['button']
                pygame.draw.rect(screen, opt_color, option_rect, border_top_left_radius=0, border_top_right_radius=0, border_bottom_left_radius=5, border_bottom_right_radius=5)
                
                opt_text_surf = FONT_SMALL.render(p_label, True, COLORS['text'])
                screen.blit(opt_text_surf, (option_rect.x + 10, option_rect.centery - opt_text_surf.get_height() / 2))
                option_y += main_rect.height

    def _draw_unified_correlation_view(self, is_global):
        if is_global:
            view_title, view_subtitle = "Global MIDI Feature Correlation", f"Dataset: All {len(self.audio_folders_data)} Files"
            correlation_data, selected_feature, selected_metric = self.global_correlation_data, self.selected_global_correlation_feature, self.selected_global_score_metric
            feature_button_rects, metric_button_rects = self.global_correlation_feature_buttons, self.global_correlation_metric_buttons
        else:
            view_title, view_subtitle = "Segment Feature Correlation", f"Piece: {self.selected_file_info.get('piece', 'N/A')}"
            correlation_data, selected_feature, selected_metric = self.correlation_data, self.selected_correlation_feature, self.selected_score_metric
            feature_button_rects, metric_button_rects = self.correlation_feature_buttons, self.correlation_metric_buttons

        LEFT_PANEL_WIDTH, PADDING = 300, 20
        PLOT_AREA_RECT = pygame.Rect(LEFT_PANEL_WIDTH + PADDING, 120, SCREEN_WIDTH - LEFT_PANEL_WIDTH - PADDING * 2, SCREEN_HEIGHT - 220)
        mouse_pos = pygame.mouse.get_pos()
        self.active_tooltip = None

        screen.blit(FONT_MAIN.render(view_title, True, COLORS['text']), (50, 15))
        screen.blit(FONT_SMALL.render(view_subtitle, True, COLORS['text']), (50, 55))
        back_button_rect = pygame.Rect(SCREEN_WIDTH - 150, 20, 130, 40)
        pygame.draw.rect(screen, COLORS['button_hover'] if back_button_rect.collidepoint(mouse_pos) else COLORS['button'], back_button_rect, border_radius=5)
        screen.blit(FONT_SMALL.render("<< Back", True, COLORS['text']), (back_button_rect.x + 30, back_button_rect.y + 10))
        
        # Define DropDown rects for click detection and drawing
        self.dropdown_rects.clear()
        dropdown_w, dropdown_h = 240, 40
        dropdown_x = back_button_rect.left - dropdown_w - 10
        self.dropdown_rects['main'] = pygame.Rect(dropdown_x, back_button_rect.y, dropdown_w, dropdown_h)

        self.correlation_controls.clear()
        feature_button_rects.clear(); pygame.draw.rect(screen, COLORS['header'], (PADDING, 100, LEFT_PANEL_WIDTH - PADDING, SCREEN_HEIGHT - 200), border_radius=5)
        list_y = 110
        sorted_features = sorted(correlation_data.items(), key=lambda item: item[1]['label'])
        for feature_key, data in sorted_features:
            is_selected = selected_feature == feature_key
            btn_rect = pygame.Rect(PADDING + 5, list_y, LEFT_PANEL_WIDTH - PADDING * 2, 35); feature_button_rects[feature_key] = btn_rect
            color = COLORS['button_hover'] if is_selected else ( (70,70,70) if not is_selected and btn_rect.collidepoint(mouse_pos) else COLORS['header'] )
            pygame.draw.rect(screen, color, btn_rect, border_radius=5)
            screen.blit(FONT_TINY.render(data['label'], True, COLORS['text']), (btn_rect.x + 10, btn_rect.y + 8)); list_y += 40

        if not selected_feature or not correlation_data.get(selected_feature):
            text_surf = FONT_MAIN.render("No data to display.", True, COLORS['text']); text_rect = text_surf.get_rect(center=PLOT_AREA_RECT.center); screen.blit(text_surf, text_rect); return

        pygame.draw.rect(screen, COLORS['piano_roll_bg'], PLOT_AREA_RECT); pygame.draw.rect(screen, COLORS['grid_line'], PLOT_AREA_RECT, 2)
        
        btn_x, btn_y = PLOT_AREA_RECT.left, PLOT_AREA_RECT.top - 45
        metric_button_rects.clear()
        for metric in ['f1', 'p', 'r']:
            label = {"f1": "F1", "p": "Precision", "r": "Recall"}[metric]; is_selected = selected_metric == metric
            btn_rect = pygame.Rect(btn_x, btn_y, 120, 35); metric_button_rects[metric] = btn_rect
            color = COLORS['button'] if not is_selected else COLORS['tp_green']
            pygame.draw.rect(screen, color, btn_rect, border_radius=5)
            text_surf = FONT_SMALL.render(label, True, COLORS['text']); text_rect = text_surf.get_rect(center=btn_rect.center); screen.blit(text_surf, text_rect); btn_x += 130
        
        checkbox_size = 20; btn_x += 20
        for key, text in [('normalize', 'Normalize (N)'), ('regression', 'Regression (R)')]:
            is_checked = getattr(self, {'normalize': 'correlation_normalize', 'regression': 'show_regression_line'}[key])
            self.correlation_controls[key] = pygame.Rect(btn_x, btn_y, 180, 35)
            cb_rect = pygame.Rect(btn_x, btn_y + (35 - checkbox_size)/2, checkbox_size, checkbox_size)
            pygame.draw.rect(screen, COLORS['checkbox_border'], cb_rect, 2, border_radius=3)
            if is_checked: pygame.draw.rect(screen, COLORS['checkbox_checked'], cb_rect.inflate(-6, -6), border_radius=3)
            screen.blit(FONT_SMALL.render(text, True, COLORS['text']), (cb_rect.right + 8, cb_rect.y - 2))
            btn_x += 180

        feature_data = correlation_data[selected_feature]; points = feature_data['points']
        if not points:
            no_points_surf = FONT_SMALL.render("No data points for this feature.", True, COLORS['text']); screen.blit(no_points_surf, no_points_surf.get_rect(center=PLOT_AREA_RECT.center)); return

        plot_feature_values = np.array([p['feature'] for p in points])
        plot_score_values = np.array([p[selected_metric] for p in points])

        if self.correlation_plot_type == 'scatter':
            self._draw_scatter_plot(PLOT_AREA_RECT, plot_feature_values, plot_score_values, points, is_global, selected_metric, feature_data['label'], feature_data['correlations'][selected_metric])
        else:
            data_to_plot = plot_feature_values if self.correlation_plot_type == 'hist_feature' else plot_score_values
            title = f"{feature_data['label']} Distribution" if self.correlation_plot_type == 'hist_feature' else f"Score ({selected_metric.upper()}) Distribution"
            self._draw_histogram(PLOT_AREA_RECT, data_to_plot, title)
            
    def _draw_scatter_plot(self, rect, f_vals, s_vals, points, is_global, metric, feature_label, corr_value):
        mouse_pos = pygame.mouse.get_pos()
        score_label_map = {"f1": "F1-Score", "p": "Precision", "r": "Recall"}

        plot_f_vals = f_vals.copy()
        plot_s_vals = s_vals.copy()

        if self.correlation_normalize:
            f_min, f_max = np.min(plot_f_vals), np.max(plot_f_vals)
            s_min, s_max = np.min(plot_s_vals), np.max(plot_s_vals)
            if f_max > f_min: plot_f_vals = (plot_f_vals - f_min) / (f_max - f_min)
            if s_max > s_min: plot_s_vals = (plot_s_vals - s_min) / (s_max - s_min)

        if self.correlation_normalize:
            min_f, max_f, min_s, max_s = 0.0, 1.0, 0.0, 1.0
            f_range, s_range = 1.0, 1.0
        else:
            min_f, max_f = np.min(plot_f_vals), np.max(plot_f_vals)
            min_s, max_s = np.min(plot_s_vals), np.max(plot_s_vals)
            f_range = (max_f - min_f) * 1.1 if max_f > min_f else 1
            s_range = (max_s - min_s) * 1.1 if max_s > min_s else 1
            min_f -= (f_range - (max_f - min_f)) / 2
            min_s -= (s_range - (max_s - min_s)) / 2

        f_range = max(f_range, 1e-9); s_range = max(s_range, 1e-9)

        AXIS_PADDING = 50; plot_w, plot_h = rect.width - AXIS_PADDING*1.5, rect.height - AXIS_PADDING*1.5
        plot_origin = (rect.left + AXIS_PADDING, rect.bottom - AXIS_PADDING)

        pygame.draw.line(screen, COLORS['grid_line'], plot_origin, (plot_origin[0], plot_origin[1] - plot_h), 2)
        pygame.draw.line(screen, COLORS['grid_line'], plot_origin, (plot_origin[0] + plot_w, plot_origin[1]), 2)
        
        for i in range(6):
            val = min_s + i/5 * s_range; y = plot_origin[1] - ((val - min_s) / s_range) * plot_h
            if plot_origin[1] - plot_h <= y <= plot_origin[1]:
                pygame.draw.line(screen, COLORS['grid_line'], (plot_origin[0] - 5, y), (plot_origin[0], y), 1)
                screen.blit(FONT_TINY.render(f"{val:.2f}", True, COLORS['text']), (plot_origin[0] - 45, y - 8))
        
        for i in range(6):
            val = min_f + i/5 * f_range; x = plot_origin[0] + ((val - min_f) / f_range) * plot_w
            if plot_origin[0] <= x <= plot_origin[0] + plot_w:
                pygame.draw.line(screen, COLORS['grid_line'], (x, plot_origin[1]), (x, plot_origin[1] + 5), 1)
                label_surf = FONT_TINY.render(f"{val:.2f}", True, COLORS['text']); label_rect = label_surf.get_rect(centerx=x, top=plot_origin[1] + 10); screen.blit(label_surf, label_rect)

        hover_info = None
        for i, point_data in enumerate(points):
            f_val, s_val = plot_f_vals[i], plot_s_vals[i]
            px = plot_origin[0] + ((f_val - min_f) / f_range) * plot_w
            py = plot_origin[1] - ((s_val - min_s) / s_range) * plot_h
            point_rect = pygame.Rect(px - 5, py - 5, 10, 10)
            if not hover_info and point_rect.collidepoint(mouse_pos):
                hover_info = {'point_data': point_data, 'pos': (px, py)}
            else:
                pygame.draw.circle(screen, COLORS['fn_yellow_fill'], (int(px), int(py)), 4)
        
        if self.show_regression_line and len(plot_f_vals) > 1:
            m, b = np.polyfit(plot_f_vals, plot_s_vals, 1)
            y1 = m * min_f + b
            y2 = m * max_f + b
            start_pos = (plot_origin[0], plot_origin[1] - ((y1 - min_s) / s_range) * plot_h)
            end_pos = (plot_origin[0] + plot_w, plot_origin[1] - ((y2 - min_s) / s_range) * plot_h)
            pygame.draw.line(screen, COLORS['regression_line'], start_pos, end_pos, 2)

        if hover_info:
            px, py = hover_info['pos']; point = hover_info['point_data']
            pygame.draw.circle(screen, COLORS['tp_green'], (int(px), int(py)), 7)
            
            score_label = score_label_map.get(metric, metric.upper())
            original_score = point[f'original_{metric}']
            tooltip_text = f"Feat: {point['feature']:.2f}, {score_label}: {original_score:.3f}"
            if not is_global: tooltip_text = f"Seg: {point['segment']}, " + tooltip_text
            else: tooltip_text = f"Piece: {point.get('piece', 'N/A')[:20]}... " + tooltip_text
            self.active_tooltip = {'text': tooltip_text, 'pos': mouse_pos}
    
    def _draw_histogram(self, rect, data, title):
        if len(data) == 0:
            no_points_surf = FONT_SMALL.render("No data for histogram.", True, COLORS['text']); screen.blit(no_points_surf, no_points_surf.get_rect(center=rect.center)); return

        counts, bin_edges = np.histogram(data, bins='auto')
        if len(counts) == 0: return

        min_val, max_val = bin_edges[0], bin_edges[-1]
        max_count = np.max(counts)
        val_range = max_val - min_val if max_val > min_val else 1
        count_range = float(max_count) if max_count > 0 else 1.0

        AXIS_PADDING = 50; plot_w, plot_h = rect.width - AXIS_PADDING*1.5, rect.height - AXIS_PADDING*1.5
        plot_origin = (rect.left + AXIS_PADDING, rect.bottom - AXIS_PADDING)
        
        screen.blit(FONT_SMALL.render(title, True, COLORS['text']), (rect.centerx - FONT_SMALL.size(title)[0]/2, rect.top + 10))

        pygame.draw.line(screen, COLORS['grid_line'], plot_origin, (plot_origin[0], plot_origin[1] - plot_h), 2)
        pygame.draw.line(screen, COLORS['grid_line'], plot_origin, (plot_origin[0] + plot_w, plot_origin[1]), 2)

        for i in range(6):
            val = i/5 * count_range; y = plot_origin[1] - (i/5 * plot_h)
            pygame.draw.line(screen, COLORS['grid_line'], (plot_origin[0] - 5, y), (plot_origin[0], y), 1)
            screen.blit(FONT_TINY.render(f"{int(val)}", True, COLORS['text']), (plot_origin[0] - 45, y - 8))
        
        for i in range(6):
            val = min_val + i/5 * val_range; x = plot_origin[0] + (i/5 * plot_w)
            pygame.draw.line(screen, COLORS['grid_line'], (x, plot_origin[1]), (x, plot_origin[1] + 5), 1)
            label_surf = FONT_TINY.render(f"{val:.2f}", True, COLORS['text']); label_rect = label_surf.get_rect(centerx=x, top=plot_origin[1] + 10); screen.blit(label_surf, label_rect)

        bar_width = (plot_w / len(counts)) * 0.9
        for i, count in enumerate(counts):
            bar_height = (count / count_range) * plot_h
            left_edge = bin_edges[i]
            bar_x = plot_origin[0] + ((left_edge - min_val) / val_range) * plot_w
            bar_rect = pygame.Rect(bar_x, plot_origin[1] - bar_height, bar_width, bar_height)
            pygame.draw.rect(screen, COLORS['gt_only_color'], bar_rect)
            pygame.draw.rect(screen, COLORS['checkbox_border'], bar_rect, 1)

if __name__ == '__main__':
    if not os.path.exists(PREDICTED_MIDI_PATH) or not os.path.exists(GROUNDTRUTH_MIDI_PATH):
        print("="*60 + "\n! ! ! CONFIGURATION ERROR ! ! !\n" + f"Cannot find PREDICTED_MIDI_PATH or GROUNDTRUTH_MIDI_PATH.\n" + "Please correctly modify the path settings at the top of the script.\n" + f"Predicted MIDI Path: {PREDICTED_MIDI_PATH}\n" + f"Ground Truth MIDI Path: {GROUNDTRUTH_MIDI_PATH}\n" + "="*60)
    else:
        app = MidiEvaluatorApp(); app.run()