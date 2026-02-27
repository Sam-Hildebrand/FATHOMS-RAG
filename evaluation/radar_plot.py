#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

# Colorblind-friendly styles
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
MARKERS     = ['o', 's', '^', 'D', 'v']


def sanitize_name(name):
    return re.sub(r"[- ./\\]+", "_", name).strip("_").replace("__", "_")


def load_nested_metrics(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    flat = {}
    
    # Extract per-category scores
    scores = data.get('scores', {})
    for cat, val in scores.items():
        flat[cat] = float(val)
    
    # Extract overall score if present
    if 'overall_score' in data:
        flat['Overall'] = float(data['overall_score'])
    elif 'overall' in data:
        flat['Overall'] = float(data['overall'])
    
    # Extract total hallucination rate if present
    if 'total_hallucination_rate' in data:
        flat['Overall_Hallucinations'] = float(data['total_hallucination_rate'])
    
    # Separate scores and hallucination rates
    score_keys = []
    hallucination_keys = []
    
    for key in scores.keys():
        if 'Hallucination Rate' in key:
            hallucination_keys.append(key)
        else:
            score_keys.append(key)
    
    # Define order: overall first, then scores in JSON order, then hallucination rates
    order = []
    if 'Overall' in flat:
        overall_items = ['Overall']
        if 'Overall_Hallucinations' in flat:
            overall_items.append('Overall_Hallucinations')
        order.append(('Overall', overall_items))
    
    if score_keys:
        order.append(('Scores', score_keys))
    
    if hallucination_keys:
        order.append(('Hallucination_Rates', hallucination_keys))
    
    return flat, order


def prepare_radar(flat_dicts, order, margin=1.0):
    # Build metric list in order, excluding hallucination rates and total hallucination rate
    metric_names = []
    for cat, benches in order:
        if cat != 'Hallucination_Rates':
            # Also exclude Total_Hallucination_Rate from Overall category
            filtered_benches = [b for b in benches if 'Hallucination' not in b]
            metric_names += filtered_benches
    raw = [[d.get(m, 0.0) for m in metric_names] for d in flat_dicts]
    arr = np.array(raw, float)
    max_v = arr.max(axis=0)
    max_v[max_v == 0] = 1.0
    norm = (arr / max_v * margin).tolist()
    return metric_names, raw, norm


def prepare_hallucination_radar(flat_dicts, order, margin=1.0):
    # Extract only hallucination-related metrics
    hallucination_names = []
    for cat, benches in order:
        if cat == 'Hallucination_Rates':
            hallucination_names += benches
        elif cat == 'Overall' and 'Overall_Hallucinations' in benches:
            hallucination_names.append('Overall_Hallucinations')
    
    if not hallucination_names:
        return None, None, None
    
    raw = [[d.get(m, 0.0) for m in hallucination_names] for d in flat_dicts]
    arr = np.array(raw, float)
    max_v = arr.max(axis=0)
    max_v[max_v == 0] = 1.0
    norm = (arr / max_v * margin).tolist()
    return hallucination_names, raw, norm


def plot_radar(metric_names,
               raw_list,
               normed_list,
               labels,
               title,
               out_dir,
               show_max_values=True,
               offset_frac=-0.08,
               filename_suffix='_radar',
               is_hallucination_plot=False):
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Adjust display names for hallucination plots to reduce overlap
    if is_hallucination_plot:
        display = []
        for m in metric_names:
            # Shorten long hallucination rate labels
            if 'Hallucination Rate' in m:
                shortened = m.replace(' Hallucination Rate', '\nHallucinations').replace('_', ' ')
            else:
                shortened = m.replace('_', '\n')
            display.append(shortened)
    else:
        display = [m.replace('_', '\n') for m in metric_names]

    # Use consistent figure size
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display, fontsize=9 if is_hallucination_plot else 10)
    ax.tick_params(axis='x', which='major', pad=30 if is_hallucination_plot else 25)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)  # leave gap beyond max

    # Plot each series with colorblind-friendly palette, styles, markers
    for i, (raw, norm, lbl) in enumerate(zip(raw_list, normed_list, labels)):
        vals = norm + [norm[0]]
        color = plt.get_cmap('tab10').colors[i % 10]
        style = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(
            angles,
            vals,
            linestyle=style,
            marker=marker,
            markersize=5,
            linewidth=2,
            color=color,
            label=lbl
        )
        ax.fill(angles, vals, alpha=0.06, color=color)

    # Add point labels for all points, avoiding duplicates and close values, with offset like original code
    for k in range(N):
        angle = angles[k]
        # Get all values for this axis with their model info
        axis_values = []
        for i, (raw, norm, lbl) in enumerate(zip(raw_list, normed_list, labels)):
            raw_val = raw[k]
            norm_val = norm[k]
            color = plt.get_cmap('tab10').colors[i % 10]
            axis_values.append((raw_val, norm_val, color, i))
        
        # Sort by raw value descending (highest first)
        axis_values.sort(key=lambda x: x[0], reverse=True)
        
        # Find max value for this axis to calculate 15% threshold
        max_val = axis_values[0][0] if axis_values else 0
        threshold = max_val * 0.15
        
        # Label values, skipping those too close to already labeled ones
        labeled_values = []
        for raw_val, norm_val, color, model_idx in axis_values:
            # Check if this value is too close to any already labeled value
            too_close = False
            for labeled_val in labeled_values:
                if abs(raw_val - labeled_val) <= threshold:
                    too_close = True
                    break
            
            if not too_close:
                labeled_values.append(raw_val)
                r_label = norm_val + 0.04
                # Apply offset like in original code
                theta = angle + offset_frac
                ax.text(theta, r_label, f'{raw_val:.2f}', fontsize=9,
                       ha='center', va='center', clip_on=False, 
                       color=color, weight='bold')

    ax.set_title(f"{title}", y=1.1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.9, -0.4))

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, sanitize_name(title) + filename_suffix + '.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved radar chart: {fname}")


def save_markdown_tables(flat_dicts, labels, order, title, out_dir):
    # Get all metrics (including hallucination rates) for the tables
    all_metric_names = []
    for cat, benches in order:
        all_metric_names += benches
    raw = [[d.get(m, 0.0) for m in all_metric_names] for d in flat_dicts]
    md = []
    
    # Overall section if present
    overall_metrics = []
    for cat, benches in order:
        if cat == 'Overall':
            overall_metrics = benches
            break
    
    if overall_metrics:
        md.append('# Overall Scores')
        header = '| Model |'
        separator = '|---|'
        for metric in overall_metrics:
            display_name = metric.replace('_', ' ')
            header += f' {display_name} |'
            separator += '---|'
        md.append(header)
        md.append(separator)
        
        for lbl, row in zip(labels, raw):
            line = f"| {lbl} |"
            for metric in overall_metrics:
                idx = all_metric_names.index(metric)
                line += f" {row[idx]:.2f} |"
            md.append(line)
        md.append('')
    
    # Detailed scores with interleaved hallucination rates
    md.append('# Detailed Scores')
    
    # Build header with scores and their corresponding hallucination rates
    detailed_headers = ['Model']
    detailed_indices = []
    
    # Get score metrics (non-hallucination, non-overall)
    score_metrics = []
    hallucination_metrics = []
    
    for cat, benches in order:
        if cat == 'Scores':
            score_metrics = benches
        elif cat == 'Hallucination_Rates':
            hallucination_metrics = benches
    
    # Pair each score with its corresponding hallucination rate
    for score_metric in score_metrics:
        detailed_headers.append(score_metric.replace('_', ' '))
        detailed_indices.append(all_metric_names.index(score_metric))
        
        # Look for corresponding hallucination rate - try multiple patterns
        potential_hallucination_names = [
            score_metric + ' Hallucination Rate',  # Most likely pattern
            score_metric + '_Hallucination_Rate',  # Alternative pattern
            score_metric + ' Hallucination_Rate',  # Another variation
        ]
        
        found_hallucination = False
        for hall_name in potential_hallucination_names:
            if hall_name in hallucination_metrics:
                detailed_headers.append(hall_name.replace('_', ' '))
                detailed_indices.append(all_metric_names.index(hall_name))
                found_hallucination = True
                break
        
        # If no exact match found, try partial matching
        if not found_hallucination:
            for hall_metric in hallucination_metrics:
                if score_metric.lower() in hall_metric.lower() and 'hallucination' in hall_metric.lower():
                    detailed_headers.append(hall_metric.replace('_', ' '))
                    detailed_indices.append(all_metric_names.index(hall_metric))
                    found_hallucination = True
                    break
    
    # Add any remaining hallucination metrics that don't have corresponding scores
    used_hallucination_metrics = set()
    for idx in detailed_indices:
        if idx < len(all_metric_names) and 'Hallucination' in all_metric_names[idx]:
            used_hallucination_metrics.add(all_metric_names[idx])
    
    for hall_metric in hallucination_metrics:
        if hall_metric not in used_hallucination_metrics:
            detailed_headers.append(hall_metric.replace('_', ' '))
            detailed_indices.append(all_metric_names.index(hall_metric))
    
    header = '| ' + ' | '.join(detailed_headers) + ' |'
    separator = '|---' + '|---' * (len(detailed_headers) - 1) + '|'
    md.append(header)
    md.append(separator)
    
    for lbl, row in zip(labels, raw):
        line_parts = [lbl]
        for idx in detailed_indices:
            line_parts.append(f"{row[idx]:.2f}")
        md.append('| ' + ' | '.join(line_parts) + ' |')
    md.append('')

    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, title + '_tables.md')
    with open(fn, 'w') as f:
        f.write('\n'.join(md))
    print(f"Saved tables: {fn}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('title')
    p.add_argument('json_files', nargs='+')
    args = p.parse_args()

    flat_all, labels, common_order = [], [], None
    for jf in args.json_files:
        flat, order = load_nested_metrics(jf)
        if common_order is None:
            common_order = order
        elif order != common_order:
            sys.exit('JSON structures differ')
        flat_all.append(flat)
        labels.append(os.path.splitext(os.path.basename(jf))[0])

    # Create main radar plot (all metrics)
    mn, raw_all, norm_all = prepare_radar(flat_all, common_order, margin=1.0)
    base = sanitize_name(args.title)
    plot_radar(mn, raw_all, norm_all, labels, args.title, base)
    
    # Create hallucination rate radar plot
    hall_names, hall_raw, hall_norm = prepare_hallucination_radar(flat_all, common_order, margin=1.0)
    if hall_names:
        hall_title = args.title + " - Hallucinations"
        plot_radar(hall_names, hall_raw, hall_norm, labels, hall_title, base, 
                  filename_suffix='_radar', is_hallucination_plot=True)
    else:
        print("No hallucination rate metrics found, skipping hallucination radar plot")
    
    # Save markdown tables
    save_markdown_tables(flat_all, labels, common_order, args.title, base)

if __name__ == '__main__':
    main()