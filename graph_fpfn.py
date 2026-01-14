import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

CROP_BAR:bool = True
CROP_YRANGE:int = 150
INJECTION_STEP:int = 12
INJECTION_TYPE:str = "FN"
INJECTION_GRAPH_STYLE:str = "line"
SEPARATE_LEARN_ADAPT:bool = False
VIS_NOSAVE:bool = True
RUN_NAME:str = "fpfn_kitsune_fp12"

MAIN_PATH:str = "./results/"
TARGET_DATASET:str = "Kitsune"

def _draw_truncated_bar(ax_bar, x, height, width, color, label, crop_limit):
    if height <= crop_limit:
        ax_bar.bar(x, height, width=width, color=color, alpha=0.6, label=label)
    else:
        left = x - width / 2
        right = x + width / 2
        bottom = 0
        top = crop_limit
        
        num_teeth = 6
        tooth_w = width / num_teeth
        tooth_h = crop_limit * 0.02
        
        verts = [(left, bottom), (left, top - tooth_h)]
        
        for i in range(num_teeth):
            verts.append((left + tooth_w * (i + 0.5), top))
            verts.append((left + tooth_w * (i + 1), top - tooth_h))
            
        verts.append((right, bottom))
        
        poly = mpatches.Polygon(verts, closed=True, facecolor=color, alpha=0.6, label=label)
        ax_bar.add_patch(poly)

def _plot():
    if SEPARATE_LEARN_ADAPT:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    data = pd.read_csv(f"{MAIN_PATH}/perf.csv")
    
    metrics = ['recall', 'precision', 'f1', 'accuracy']
    colors = {'recall': 'green', 'precision': 'orange', 'f1': 'blue', 'accuracy': 'purple'}
    
    ax_bar = ax.twinx()
    
    ax_inj = ax.twinx()
    ax_inj.spines["right"].set_position(("axes", 1.08))
    ax_inj.set_frame_on(True)
    ax_inj.patch.set_visible(False)
    
    x_pos = 0
    x_labels = []
    x_ticks = []
    
    min_turn = data['turn'].min()
    
    bar_width = 0.35
    
    cumulative_inj = 0
    inj_x = []
    inj_y = []
    plotted_labels = set()
    
    for idx, row in data.iterrows():
        for metric in metrics:
            entry_val = row[f'entry_{metric}']
            exit_val = row[f'exit_{metric}']
            
            ax.plot([x_pos, x_pos + 1], [entry_val, exit_val], 
                   color=colors[metric], marker='o', label=metric if idx == 0 else "")
            
            if idx < len(data) - 1:
                next_entry_val = data.iloc[idx + 1][f'entry_{metric}']
                ax.plot([x_pos + 1, x_pos + 2], [exit_val, next_entry_val], 
                       color=colors[metric], linestyle=':', marker='o')
        
        target_bars = []
        
        if INJECTION_GRAPH_STYLE == 'bar':
            if INJECTION_TYPE == "FN":
                candidates = [
                    ('fn_forced_removed', 'skyblue', 'Forced Removed'),
                    ('fn_recovered_delta', 'salmon', 'Recovered Delta'),
                    ('fn_recovered_total', 'orange', 'Recovered Total'),
                    ('fn_removed', 'gray', 'FN Removed')
                ]
            else:
                candidates = [
                    ('fp_injected', 'skyblue', 'Injected'),
                    ('fp_removed', 'salmon', 'Removed')
                ]
            
            for col, color, lbl in candidates:
                val = row.get(col, 0)
                if val > 0:
                    target_bars.append((val, color, lbl))
        else:
            generated = row['generated']
            removed = row['removed']
            
            target_bars = [(generated, 'skyblue', 'Generated'), (removed, 'salmon', 'Removed')]
            
            inj_x.append(x_pos)
            inj_y.append(cumulative_inj)
            
            if row['turn'] >= INJECTION_STEP:
                if INJECTION_TYPE == "FN":
                    cumulative_inj += (row['fn_forced_removed'] - row['fn_recovered_delta'])
                else:
                    cumulative_inj += (row['fp_injected'] - row['fp_removed'])
                if cumulative_inj < 0:
                    cumulative_inj = 0
            
            inj_x.append(x_pos + 1)
            inj_y.append(cumulative_inj)
        
        bar_x_center = x_pos + 0.5
        num_bars = len(target_bars)
        
        if num_bars > 0:
            total_width = 0.7
            w = total_width / num_bars
            start_x = bar_x_center - (total_width / 2) + (w / 2)
            
            for i, (val, color, lbl) in enumerate(target_bars):
                bx = start_x + i * w
                
                final_lbl = lbl if lbl not in plotted_labels else ""
                if lbl not in plotted_labels:
                    plotted_labels.add(lbl)
                
                if CROP_BAR and CROP_YRANGE > 0:
                    _draw_truncated_bar(ax_bar, bx, val, w, color, final_lbl, CROP_YRANGE)
                    ax_bar.text(bx, min(val, CROP_YRANGE) + 2, 
                               str(int(val)), ha='center', va='bottom', fontsize=8, rotation=45)
                else:
                    ax_bar.bar(bx, val, width=w, color=color, alpha=0.6, label=final_lbl)
        
        x_labels.extend([f'{int(row["turn"])}-entry', f'{int(row["turn"])}-exit'])
        x_ticks.extend([x_pos, x_pos + 1])
        x_pos += 2
    
    injection_x = (INJECTION_STEP - min_turn) * 2
    vline_label = "Removal" if INJECTION_TYPE == "FN" else "Injection"
    ax.axvline(x=injection_x, color='red', linestyle=':', linewidth=2, label=vline_label)
    
    ax.set_xlabel('Turn Phase')
    ax.set_ylabel('Metric Value')
    ax_bar.set_ylabel('Generated/Removed Count')
    
    if INJECTION_GRAPH_STYLE == 'line':
        injection_start_x = (INJECTION_STEP - min_turn) * 2
        final_inj_x = [x for x in inj_x if x >= injection_start_x]
        final_inj_y = [y for x, y in zip(inj_x, inj_y) if x >= injection_start_x]
        
        if final_inj_x:
            inj_label = "Removed FN Count" if INJECTION_TYPE == "FN" else f"Injected {INJECTION_TYPE} Count"
            ax_inj.plot(final_inj_x, final_inj_y, color='red', linewidth=2, label=inj_label, alpha=0.5)
            ax_inj.fill_between(final_inj_x, final_inj_y, color='red', alpha=0.1)
            ax_inj.set_ylabel(inj_label, color='red')
            ax_inj.tick_params(axis='y', labelcolor='red')
        else:
            ax_inj.set_visible(False)
    else:
        ax_inj.set_visible(False)
    
    if CROP_BAR and CROP_YRANGE > 0:
        ax_bar.set_ylim(0, CROP_YRANGE)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_bar.get_legend_handles_labels()
    lines3, labels3 = ax_inj.get_legend_handles_labels()
    
    # Reorder so Injection and Injected Count are last
    main_lines = [l for l, lbl in zip(lines1, labels1) if lbl != vline_label]
    main_labels = [lbl for lbl in labels1 if lbl != vline_label]
    inj_lines = [l for l, lbl in zip(lines1, labels1) if lbl == vline_label]
    inj_labels = [lbl for lbl in labels1 if lbl == vline_label]
    
    all_lines = main_lines + lines2 + inj_lines + lines3
    all_labels = [l[0].upper() + l[1:] if l else l for l in main_labels + labels2 + inj_labels + labels3]
    ax.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.15, 1))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    if VIS_NOSAVE:
        plt.show()
    else:
        os.makedirs("graph", exist_ok=True)
        plt.savefig(f"graph/{RUN_NAME}.png")
    
if __name__ == "__main__":
    _plot()
