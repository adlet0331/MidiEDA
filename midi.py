# [SOURCE] https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/midi.py

import multiprocessing
import sys

import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi
from tqdm import tqdm

def load_midi(path) -> np.ndarray:
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:]
                          if n['type'] == 'sustain_off' or n['note'] == onset['note'] or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)

def slice_midi(pitches_hz, intervals_sec, velocities, num_segments, segment_length_sec) -> list:
    returned_segments = []
    for i in range(num_segments):
        segment_start_time = i * segment_length_sec
        segment_end_time = (i + 1) * segment_length_sec
        
        segment_pitches, segment_intervals, segment_velocities = [], [], []
        
        for note_idx in range(len(pitches_hz)):
            note_start_time = intervals_sec[note_idx, 0]
            note_end_time = intervals_sec[note_idx, 1]
            
            if note_start_time < segment_end_time and note_end_time > segment_start_time:
                adjusted_interval = [
                    max(note_start_time - segment_start_time, 0),
                    min(note_end_time - segment_start_time, segment_length_sec)
                ]
                if adjusted_interval[0] < adjusted_interval[1]:
                    segment_pitches.append(pitches_hz[note_idx])
                    segment_intervals.append(adjusted_interval)
                    segment_velocities.append(velocities[note_idx])

        returned_segments.append({
            'pitches': np.array(segment_pitches),
            'intervals': np.array(segment_intervals),
            'velocities': np.array(segment_velocities)
        })

    return returned_segments

def save_midi(path, pitches, intervals, velocities) -> None:
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)

import numpy as np

def notes_to_piano_roll(notes, fs=120, max_time=None):
    """
    Converts notes lists to a piano roll (numpy array).

    Args:
        notes (list): A list of note objects containing pitch, start, end, and velocity.
        fs (int): The sampling rate for the piano roll (frames per second).
        max_time (float, optional): The maximum time for the piano roll. 
                                    If None, it's determined by the longest note.

    Returns:
        np.ndarray: A 2D numpy array representing the piano roll.
                    Rows are pitches (0-127), columns are time steps.
    """
    if not notes:
        return np.array([])

    if max_time is None:
        max_time = max(note.end for note in notes)

    # Determine the number of time steps
    n_steps = int(np.ceil(max_time * fs))
    
    # Initialize piano roll with zeros (no notes playing)
    piano_roll = np.zeros((128, n_steps), dtype=np.uint8) 

    for note in notes:
        start_idx = int(note.start * fs)
        end_idx = int(note.end * fs)
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(n_steps, end_idx) 
        
        # Set the velocity for the duration of the note
        piano_roll[int(note.pitch), start_idx:end_idx] = int(note.velocity)
        
    return piano_roll