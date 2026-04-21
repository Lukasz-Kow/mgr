from tensorboard.backend.event_processing import event_accumulator
import os

def extract_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    results = {}
    for tag in tags:
        events = ea.Scalars(tag)
        results[tag] = events[-1].value # Get last value
    return results

log_roots = ['runs/baseline']
# Find subdirs if any
for root in log_roots:
    if os.path.exists(root):
        print(f"--- Metrics for {root} ---")
        try:
            metrics = extract_metrics(root)
            for tag, val in metrics.items():
                print(f"{tag}: {val:.4f}")
        except Exception as e:
            # Try subdirectories
            for sub in os.listdir(root):
                sub_path = os.path.join(root, sub)
                if os.path.isdir(sub_path):
                    print(f"  --- {sub} ---")
                    try:
                        metrics = extract_metrics(sub_path)
                        for tag, val in metrics.items():
                            print(f"    {tag}: {val:.4f}")
                    except: pass
    else:
        print(f"Log dir {root} not found.")
