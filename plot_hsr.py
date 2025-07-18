#!/usr/bin/env python3
# Copyright 2025 raider, help from lea_calot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import random
import os
import sys
from collections import defaultdict

def get_mode_color(mode_id, color_map={}):
    """
    Generates a consistent, random color for a given mode ID.

    Parameters:
        mode_id: The identifier for the mode.
        color_map (dict, optional): A dictionary mapping mode IDs to color hex strings.
            Used to cache and ensure consistent colors across calls. Defaults to an empty dict.

    Returns:
        str: Hex string representing the color for the given mode ID.
    """
    if mode_id not in color_map:
        random.seed(str(mode_id))
        color_map[mode_id] = mcolors.to_hex([random.random(), random.random(), random.random()])
    return color_map[mode_id]

def process_stimulation_data(data):
    """
    Processes raw stimulation session data and builds plot-ready structures.

    Parameters:
        data (dict): The session data from the JSON file, containing metadata, initial context, and events.

    Returns:
        dict: A state dictionary with processed plot data, boost/shock logic, modes, and fire events.
    """
    # --- State Initialization ---
    state = {
        'g1': {'base_intensity': 0, 'plot_data': defaultdict(list), 'adjust_value': 0, 'adjust_plot_data': defaultdict(list)},
        'g2': {'base_intensity': 0, 'plot_data': defaultdict(list), 'adjust_value': 0, 'adjust_plot_data': defaultdict(list)},
        'boost': {
            'shockMode': False,
            'channels': [False, False],
            'duration': 0.1,
            'offset': 0
        },
        'modes': {
            'g1': [],
            'g2': []
        },
        'fire_events': []
    }

    # Load initial context
    initial_context = data.get('initialContext', {})
    state['g1']['base_intensity'] = initial_context.get('generator1', {}).get('intensity', 0)
    state['g2']['base_intensity'] = initial_context.get('generator2', {}).get('intensity', 0)

    state['g1']['adjust_value'] = initial_context.get('generator1', {}).get('adjust', 50) # Default adjust is 50
    state['g2']['adjust_value'] = initial_context.get('generator2', {}).get('adjust', 50) # Default adjust is 50

    # Initialize plot data at t=0
    state['g1']['plot_data']['time'].append(0)
    state['g1']['plot_data']['value'].append(state['g1']['base_intensity'])
    state['g2']['plot_data']['time'].append(0)
    state['g2']['plot_data']['value'].append(state['g2']['base_intensity'])

    state['g1']['adjust_plot_data']['time'].append(0)
    state['g1']['adjust_plot_data']['value'].append(state['g1']['adjust_value'])
    state['g2']['adjust_plot_data']['time'].append(0)
    state['g2']['adjust_plot_data']['value'].append(state['g2']['adjust_value'])

    # Initialize boost state
    initial_boost = initial_context.get('boost', {})
    for key in state['boost']:
        if key in initial_boost:
            state['boost'][key] = initial_boost[key]

    # Initialize modes
    g1_initial_mode = initial_context.get('generator1', {}).get('mode')
    g2_initial_mode = initial_context.get('generator2', {}).get('mode')
    if g1_initial_mode:
        state['modes']['g1'].append({'ts': 0, 'mode': g1_initial_mode})
    if g2_initial_mode:
        state['modes']['g2'].append({'ts': 0, 'mode': g2_initial_mode})

    # --- Event Processing Loop ---
    # Ensure events are sorted by timestamp
    all_events = sorted(data.get('data', []), key=lambda x: x.get('ts', 0))

    # Keep track of the last known boost channels state for fire events
    current_boost_channels = list(state['boost']['channels']) # Make a copy

    for event in all_events:
        ts = event.get('ts')
        path = event.get('path', [])
        value = event.get('value')
        event_type = event.get('type')

        if ts is None or not path:
            continue

        # --- Handle Boost State Changes ---
        if path[0] == 'boost':
            prop = path[1]
            if prop == 'channels':
                current_boost_channels = list(value) # Update the current channels state
                state['boost']['channels'] = value # Also update the main state for other logic
            elif prop in state['boost']:
                # Special handling for shockMode state changes
                if prop == 'shockMode' and state['boost']['shockMode'] != value:
                    state['boost']['shockMode'] = value
                    is_entering_shock = value

                    for i, gen_key in enumerate(['g1', 'g2']):
                        if state['boost']['channels'][i]: # If this channel is affected
                            if is_entering_shock:
                                # For shock mode enable drop intensity to 0
                                state[gen_key]['plot_data']['time'].append(ts)
                                state[gen_key]['plot_data']['value'].append(0)
                            else:
                                # Restore base intensity after
                                state[gen_key]['plot_data']['time'].append(ts)
                                state[gen_key]['plot_data']['value'].append(state[gen_key]['base_intensity'])
                else:
                    state['boost'][prop] = value

            # --- Handle Fire Events ---
            elif prop == 'fire' and event_type == 'action':
                # Store fire event with the boost channels state at that time
                state['fire_events'].append({'ts': ts, 'channels': list(current_boost_channels)})

                for i, gen_key in enumerate(['g1', 'g2']):
                    if state['boost']['channels'][i]: # If this channel is boosted
                        if state['boost']['shockMode']:
                            # Shock Pulse: Go to base intensity for a duration, then back to 0
                            duration_ms = state['boost']['duration'] * 1000
                            state[gen_key]['plot_data']['time'].append(ts)
                            state[gen_key]['plot_data']['value'].append(state[gen_key]['base_intensity'])
                            state[gen_key]['plot_data']['time'].append(ts + duration_ms)
                            state[gen_key]['plot_data']['value'].append(0)
                        else:
                            # Normal Boost: Pulse with offset for 1ms
                            # Ensure the boosted value does not exceed 100
                            boosted_val = min(100, state[gen_key]['base_intensity'] + state['boost']['offset'])
                            current_val = state[gen_key]['plot_data']['value'][-1]
                            state[gen_key]['plot_data']['time'].append(ts)
                            state[gen_key]['plot_data']['value'].append(boosted_val)
                            state[gen_key]['plot_data']['time'].append(ts + 1)
                            state[gen_key]['plot_data']['value'].append(current_val)

        # --- Handle Generator State Changes ---
        elif path[0] in ['generator1', 'generator2']:
            gen_key = 'g1' if path[0] == 'generator1' else 'g2'
            prop = path[1]
            if prop == 'intensity':
                state[gen_key]['base_intensity'] = value
                # If not in shock mode for this channel, update plot. Otherwise, just update the base.
                is_in_shock = state['boost']['shockMode'] and state['boost']['channels'][0 if gen_key == 'g1' else 1]
                if not is_in_shock:
                    state[gen_key]['plot_data']['time'].append(ts)
                    state[gen_key]['plot_data']['value'].append(value)
            elif prop == 'mode':
                state['modes'][gen_key].append({'ts': ts, 'mode': value})
            elif prop == 'adjust':
                state[gen_key]['adjust_value'] = value
                state[gen_key]['adjust_plot_data']['time'].append(ts)
                state[gen_key]['adjust_plot_data']['value'].append(value)


    # Determine true session end for extending adjust plot data
    max_ts_in_data = 0
    if data.get('data'):
        max_ts_in_data = max(event.get('ts', 0) for event in data['data'])
    total_duration_ms = max(data.get('metadata', {}).get('duration', 0), max_ts_in_data)

    # Extend adjust plot data to the end of the session
    for gen_key in ['g1', 'g2']:
        if state[gen_key]['adjust_plot_data']['time'] and state[gen_key]['adjust_plot_data']['time'][-1] < total_duration_ms:
            state[gen_key]['adjust_plot_data']['time'].append(total_duration_ms)
            state[gen_key]['adjust_plot_data']['value'].append(state[gen_key]['adjust_value'])


    # Sort all plot data by time to ensure correctness
    for gen_key in ['g1', 'g2']:
        time_data = state[gen_key]['plot_data']['time']
        value_data = state[gen_key]['plot_data']['value']
        sorted_pairs = sorted(zip(time_data, value_data))
        if sorted_pairs:
            state[gen_key]['plot_data']['time'], state[gen_key]['plot_data']['value'] = zip(*sorted_pairs)
        else:
            state[gen_key]['plot_data']['time'], state[gen_key]['plot_data']['value'] = [], []

        time_data_adj = state[gen_key]['adjust_plot_data']['time']
        value_data_adj = state[gen_key]['adjust_plot_data']['value']
        sorted_pairs_adj = sorted(zip(time_data_adj, value_data_adj))
        if sorted_pairs_adj:
            state[gen_key]['adjust_plot_data']['time'], state[gen_key]['adjust_plot_data']['value'] = zip(*sorted_pairs_adj)
        else:
            state[gen_key]['adjust_plot_data']['time'], state[gen_key]['adjust_plot_data']['value'] = [], []

    return state


def plot_stimulation_data(file_path, output_path='output.png'):
    """
    Loads stimulation session data from a JSON file, processes it, and generates a plot.

    Parameters:
        file_path (str): Path to the input JSON file containing session data.
        output_path (str, optional): Path to save the generated plot image. Defaults to 'output.png'.

    Returns:
        None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file: {e}")
        return

    # --- Process Data ---
    state = process_stimulation_data(data)
    mode_map = {mode['id']: mode['title'] for mode in data.get('modes', [])}

    # Determine true session end (already calculated in process_stimulation_data, but for plotting context)
    max_ts_in_data = 0
    if data.get('data'):
        max_ts_in_data = max(event.get('ts', 0) for event in data['data'])
    total_duration_ms = max(data.get('metadata', {}).get('duration', 0), max_ts_in_data)


    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    basename = os.path.basename(file_path)
    fig.suptitle(f"{basename}", fontsize=16)

    # Plot main intensity lines in green
    ax1.plot(state['g1']['plot_data']['time'], state['g1']['plot_data']['value'], label='Intensity', color='green', drawstyle='steps-post')
    ax2.plot(state['g2']['plot_data']['time'], state['g2']['plot_data']['value'], label='Intensity', color='green', drawstyle='steps-post')

    # Plot adjust lines in yellow
    ax1.plot(state['g1']['adjust_plot_data']['time'], state['g1']['adjust_plot_data']['value'], label='Adjust', color='yellow', drawstyle='steps-post')
    ax2.plot(state['g2']['adjust_plot_data']['time'], state['g2']['adjust_plot_data']['value'], label='Adjust', color='yellow', drawstyle='steps-post')

    # Plot fire event markers
    for fire_event in state['fire_events']:
        ts = fire_event['ts']
        channels_at_fire = fire_event['channels']

        if channels_at_fire[0]: # If Generator 1 channel was enabled during this fire event
            ax1.text(ts, ax1.get_ylim()[1] * 0.9, '⚡', color='orange', fontsize=18, ha='center', va='top', fontweight='bold')
        if channels_at_fire[1]: # If Generator 2 channel was enabled during this fire event
            ax2.text(ts, ax2.get_ylim()[1] * 0.9, '⚡', color='orange', fontsize=18, ha='center', va='top', fontweight='bold')

    # Custom y-axis formatter to hide labels below 0
    def format_yaxis_labels(y, pos):
        if y < 0:
            return ''
        return f'{int(y)}'

    # --- Draw Timeline Bars ---
    rect_height = 5
    # Adjusted rect_y to place mode bars in the [-10, -1] range
    rect_y = -6

    for i, (ax, gen_key) in enumerate([(ax1, 'g1'), (ax2, 'g2')]):
        # Removed ax.set_ylabel('Intensity')
        ax.grid(True)
        # Set y-axis limits to [-10, 100]
        ax.set_ylim(-10, 100)
        # Apply custom y-axis formatter
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis_labels))

        for j in range(len(state['modes'][gen_key])):
            start_ts = state['modes'][gen_key][j]['ts']
            end_ts = state['modes'][gen_key][j+1]['ts'] if j + 1 < len(state['modes'][gen_key]) else total_duration_ms
            duration_ts = end_ts - start_ts
            if duration_ts <= 0: continue

            mode_id = state['modes'][gen_key][j]['mode']
            mode_title = mode_map.get(mode_id, 'Unknown')
            color = get_mode_color(mode_id)
            ax.broken_barh([(start_ts, duration_ts)], (rect_y, rect_height), facecolors=color, alpha=0.4)
            # Adjust text position to be centered within the new rect_y
            ax.text(start_ts + duration_ts / 2, rect_y + rect_height / 2, mode_title, ha='center', va='center', color='black', fontsize=8, alpha=0.7)

    # Consolidated legend on ax1
    ax1.legend(loc='upper right')

    # --- Axis Formatting and Final Touches ---
    ax2.set_xlabel('Time (hh:mm:ss)')
    def format_ms_to_hhmmss(x, pos):
        total_seconds = int(x // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_ms_to_hhmmss))
    ax2.set_xlim(0, total_duration_ms)

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_hsr.py <path_to_json_file> <output_image_path.png>")
        sys.exit(1)
    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    plot_stimulation_data(json_file_path, output_file_path)