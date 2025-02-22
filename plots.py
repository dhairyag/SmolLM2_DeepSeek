import torch
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

def load_metrics_from_checkpoints():
    """Load metrics from all checkpoint files"""
    metrics_files = glob.glob('checkpoints/metrics_*.pt')
    
    # Dictionary to store all metrics
    all_metrics = {
        'steps': [],
        'loss': [],
        'unique_content': []
    }
    
    # Sort files by step number
    def get_step_num(filename):
        match = re.search(r'metrics_(\d+|final|additional).pt', filename)
        if match:
            if match.group(1) == 'final':
                return float('inf')-1
            elif match.group(1) == 'additional':
                return float('inf')
            return int(match.group(1))
        return 0
        
    metrics_files.sort(key=get_step_num)
    
    # Load each checkpoint
    for metric_file in metrics_files:
        try:
            # Load with weights_only=True for security
            metrics = torch.load(metric_file, weights_only=True)
            
            # Extract step number from filename
            step_match = re.search(r'metrics_(\d+|final|additional).pt', metric_file)
            if step_match:
                step = step_match.group(1)
                if step == 'final':
                    step = 10000
                elif step == 'additional':
                    step = 10050
                else:
                    step = int(step)
                
                # Get the last loss value from the list for this checkpoint
                if isinstance(metrics['loss'], list):
                    loss = metrics['loss'][-1]  # Take last loss value
                else:
                    loss = metrics['loss']  # If it's a single value
                
                # Get unique content count if available
                unique_count = metrics.get('unique_content_count', [0])[-1] if isinstance(metrics.get('unique_content_count', [0]), list) else metrics.get('unique_content_count', 0)
                
                all_metrics['steps'].append(step)
                all_metrics['loss'].append(loss)
                all_metrics['unique_content'].append(unique_count)
                
        except Exception as e:
            print(f"Error loading {metric_file}: {e}")
            
    # Sort metrics by steps
    sorted_indices = np.argsort(all_metrics['steps'])
    all_metrics['steps'] = np.array(all_metrics['steps'])[sorted_indices]
    all_metrics['loss'] = np.array(all_metrics['loss'])[sorted_indices]
    all_metrics['unique_content'] = np.array(all_metrics['unique_content'])[sorted_indices]
    
    return all_metrics

def create_plots():
    """Create and save training metric plots"""
    metrics = load_metrics_from_checkpoints()
    
    # Set bigger font sizes globally
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot 1: Loss vs Steps
    ax1.plot(metrics['steps'], metrics['loss'], 'b.-', linewidth=2, markersize=8,
             label='Training Loss')
    ax1.set_title('Training Loss vs Steps', pad=20)
    ax1.set_xlabel('Steps', labelpad=12)
    ax1.set_ylabel('Loss', labelpad=12)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.legend(fontsize=14)
    ax1.set_yscale('log')
    # Make tick labels bigger
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: Unique Content vs Steps
    ax2.plot(metrics['steps'], metrics['unique_content'], 'r.-', linewidth=2, 
             markersize=8, label='Unique Content')
    ax2.set_title('Unique Content vs Steps', pad=20)
    ax2.set_xlabel('Steps', labelpad=12)
    ax2.set_ylabel('Unique Content Count', labelpad=12)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.legend(fontsize=14)
    # Make tick labels bigger
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Add more padding between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Save plots with higher quality
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_plots()
