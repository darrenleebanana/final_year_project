import json
from collections import defaultdict
import os
import cv2
import time
import pandas as pd

fps = 24
name = "10s"
BASE_DIR = f"image_database/classroom"
timestep_json = os.path.join(BASE_DIR, f"{name}_individual_timestep.json")
processed_json = os.path.join(BASE_DIR, f"{name}_demographics.json") 
cluster_json = os.path.join(BASE_DIR, f"{name}_cluster.json")
track_across_cluster = os.path.join(BASE_DIR, f"{name}_cluster_timestep.json")

original_video = os.path.join(BASE_DIR, f"{name}.mp4")
individual_output_video = os.path.join(BASE_DIR, f"{name}_individual.mp4")
cluster_output_video = os.path.join(BASE_DIR, f"{name}_cluster.mp4")
excel_output = os.path.join(BASE_DIR, f"{name}_cluster.xlsx")

TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
FONT_SCALE, FONT_THICKNESS = 0.7, 1
IOU_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 60

# Function to calculate IoU
def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def compute_tracking_statistics(timestep_json, processed_json, IOU_THRESHOLD, fps):
    """Group IDs based on bounding box similarity using IoU and track statistics."""
    with open(timestep_json, "r") as f:
        data = json.load(f)

    total_frames = len(data)  # Count total frames
    frequency_table = {}  # Tracks frame appearances per ID
    longest_consecutive_presence = {}  # Stores longest continuous presence
    current_streak = {}  # Tracks ongoing streaks
    last_seen_frame = {}  # Tracks last frame where an ID was seen
    grouped_ids = {}  # Dictionary for grouped IDs
    id_to_group = {}  # Maps each ID to assigned group

    for frame_key, frame_objects in data.items():
        frame_number = int(frame_key.split("_")[1])  # Extract frame number

        for obj in frame_objects:
            track_id = obj["id"]
            if track_id in excluded_ids:
                continue
            bbox = obj["bbox"]

            # Track consecutive presence
            if track_id in last_seen_frame and last_seen_frame[track_id] == frame_number - 1:
                current_streak[track_id] += 1  # Continue streak
            else:
                current_streak[track_id] = 1  # Reset streak

            # Update last seen frame
            last_seen_frame[track_id] = frame_number
            frequency_table[track_id] = frequency_table.get(track_id, 0) + 1

            # Update longest consecutive presence
            longest_consecutive_presence[track_id] = max(
                longest_consecutive_presence.get(track_id, 0),
                current_streak[track_id]
            )

            # Group IDs based on IoU
            found_group = None
            for group_id, members in grouped_ids.items():
                for member_id in members:
                    if member_id in id_to_group:
                        existing_bbox = id_to_group[member_id]["bbox"]
                        iou = calculate_iou(existing_bbox, bbox)
                        if iou > IOU_THRESHOLD:  # If IoU is high, they belong to the same group
                            found_group = group_id
                            break
                if found_group:
                    break
            
            if found_group:
                grouped_ids[found_group].add(track_id)  # Add ID to existing group
                id_to_group[track_id] = {"bbox": bbox, "group": found_group}
            else:
                grouped_ids[track_id] = {track_id}  # Create new group
                id_to_group[track_id] = {"bbox": bbox, "group": track_id}

    # Calculate additional statistics
    final_results = {}
    for track_id, frames_present in frequency_table.items():
        total_duration = frames_present / fps  # Convert frames to seconds
        presence_percentage = (frames_present / total_frames) * 100
        longest_consec_duration = longest_consecutive_presence.get(track_id, 0) / fps

        final_results[track_id] = {
            "Total Frames": frames_present,
            "Total Frames (%)": round(presence_percentage, 2),
            "Total Duration (s)": round(total_duration, 2),
            "Longest Consecutive Frames": longest_consecutive_presence.get(track_id, 0),
            "Longest Consecutive Duration (s)": round(longest_consec_duration, 2)
        }

    # Save results
    results = {
        "grouped_ids": {group: list(members) for group, members in grouped_ids.items()},
        "tracking_statistics": final_results
    }

    with open(processed_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processed tracking results saved as {processed_json}")
    
    return len(results["grouped_ids"]), total_frames
    
def group_and_merge_clusters(processed_json, cluster_json):
    # --- Load JSON File ---
    with open(processed_json, "r") as f:
        data = json.load(f)

    # Extract data
    tracking_statistics = data["tracking_statistics"]
    grouped_ids = data["grouped_ids"]

    # Remove excluded IDs from tracking statistics
    tracking_statistics = {k: v for k, v in tracking_statistics.items() if k not in excluded_ids}

    # Remove excluded IDs from grouped_ids and build a mapping from ID to group(s)
    filtered_grouped_ids = {}
    id_group_map = defaultdict(set)

    for group_id, member_ids in grouped_ids.items():
        filtered_members = {id_ for id_ in member_ids if str(id_) not in excluded_ids}
        if filtered_members:
            filtered_grouped_ids[group_id] = filtered_members
            for id_ in filtered_members:
                id_group_map[str(id_)].add(group_id)

    # Identify IDs appearing in multiple groups
    initial_multi_group_ids = {id_: groups for id_, groups in id_group_map.items() if len(groups) > 1}

    # Remove single-ID groups if the ID is already in a larger group
    to_remove = set()
    for id_, groups in initial_multi_group_ids.items():
        for group_id in groups:
            if len(filtered_grouped_ids[group_id]) == 1:
                to_remove.add(group_id)

    # Remove identified single-ID groups
    filtered_grouped_ids = {k: v for k, v in filtered_grouped_ids.items() if k not in to_remove}

    # Recompute multi-group IDs after dropping non-essential groupings
    final_id_group_map = defaultdict(set)
    for group_id, member_ids in filtered_grouped_ids.items():
        for id_ in member_ids:
            final_id_group_map[str(id_)].add(group_id)

    # Only consider IDs that now appear in multiple groups
    multi_group_ids = {id_: groups for id_, groups in final_id_group_map.items() if len(groups) > 1}

    # Process grouped IDs: merge the statistics for each group
    merged_stats = {}

    for group_id, member_ids in filtered_grouped_ids.items():
        combined_stats = {
            "Total Frames": 0,
            "Total Frames (%)": 0,
            "Total Duration (s)": 0,
            "Longest Consecutive Frames": 0,
            "Longest Consecutive Duration (s)": 0,
            "Cluster Size": len(member_ids),
            "IDs in Cluster": ", ".join(str(id_) for id_ in sorted(member_ids))
        }

        # Aggregate statistics for each ID in the group
        for member_id in member_ids:
            member_id_str = str(member_id)
            if member_id_str in tracking_statistics:
                stats = tracking_statistics[member_id_str]
                combined_stats["Total Frames"] += stats["Total Frames"]
                combined_stats["Total Duration (s)"] += stats["Total Duration (s)"]
                combined_stats["Longest Consecutive Frames"] = max(
                    combined_stats["Longest Consecutive Frames"], stats["Longest Consecutive Frames"]
                )
                combined_stats["Longest Consecutive Duration (s)"] = max(
                    combined_stats["Longest Consecutive Duration (s)"], stats["Longest Consecutive Duration (s)"]
                )
                combined_stats["Total Frames (%)"] += stats["Total Frames (%)"]

        # Store in merged_stats
        merged_stats[group_id] = combined_stats

    # Sort merged_stats by numerical group ID
    sorted_merged_stats = {k: merged_stats[k] for k in sorted(merged_stats.keys(), key=int)}

    # Save only Cluster ID and associated IDs to a separate JSON
    grouped_cluster_ids = [
        {"Cluster ID": k, "IDs in Cluster": v["IDs in Cluster"]}
        for k, v in zip(range(len(sorted_merged_stats)), sorted_merged_stats.values())
    ]

    with open(cluster_json, "w") as f:
        json.dump(grouped_cluster_ids, f, indent=4)

    print(f"Cluster ID list saved to {cluster_json}")

    # Display IDs appearing in multiple groups
    print("\n=== IDs Appearing in Multiple Groups (After Dropping Non-Essential Groupings) ===")
    if multi_group_ids:
        for id_, groups in multi_group_ids.items():
            print(f"ID {id_} appears in groups: {sorted(groups)}")
    else:
        print("No IDs appear in multiple groups.")

def assign_clusters_to_detections(timestep_json, cluster_json, track_across_cluster):

    # --- Load JSON Files ---
    with open(timestep_json, "r") as f:
        tracking_data = json.load(f)

    with open(cluster_json) as f:
        cluster_list = json.load(f)  # This is a list of dictionaries

    # --- Create ID-to-Cluster Mapping ---
    id_to_cluster = {}
    for cluster in cluster_list:
        cluster_id = f"Cluster {cluster['Cluster ID']}"  # Convert to consistent string format
        ids = cluster["IDs in Cluster"].split(", ")  # Convert comma-separated string into a list
        for id_ in ids:
            id_to_cluster[id_] = cluster_id  # Assign each ID to its cluster

    # --- Aggregate Data Per Cluster & Timestamp ---
    clustered_results = defaultdict(list)  # {cluster: [{Frame, Emotion, Confidence_Scores}]}

    for frame, detections in tracking_data.items():
        frame_number = int(frame.split("_")[1])  # Extract numerical frame number

        for obj in detections:
            track_id = str(obj["id"])  # Convert ID to string
            emotion = obj["emotion"]
            bbox = obj["bbox"]
            confidence_scores = obj["confidence_scores"]

            # Assign this data to the corresponding cluster
            if track_id in id_to_cluster:
                cluster_id = id_to_cluster[track_id]

                clustered_results[cluster_id].append({
                    "frame": frame_number,
                    "individual_id": track_id,
                    "bbox": bbox,
                    "emotion": emotion,
                    "confidence_scores": confidence_scores
                })
                
    # --- Save Results to JSON ---
    with open(track_across_cluster, "w") as f:
        json.dump(clustered_results, f, indent=4)

    print(f"Clustered emotion timeline saved to {track_across_cluster}")
    
def annotate_video_cluster(original_video, json_path, output_video):
    # Load JSON Data
    with open(json_path, "r") as f:
        clustered_data = json.load(f)

    # Open video file
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    # Read frames and annotate
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        frame_number += 1

        # Draw annotations for each cluster at the current frame
        for cluster_id, detections in clustered_data.items():
            for obj in detections:
                if obj["frame"] == frame_number:
                    bbox = obj["bbox"]
                    x1, y1, x2, y2 = bbox
                    emotion = obj["emotion"]
                    confidence_scores = obj["confidence_scores"]
                   
                    # Determine confidence value
                    confidence = confidence_scores.get(emotion, 0)
                    text = f"{emotion} ({confidence:.1f}%)" if confidence >= CONFIDENCE_THRESHOLD else "None"

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), FONT_THICKNESS)

                    # Display Cluster ID (Above Bbox)
                    cv2.putText(frame, f"{cluster_id.split()[1]}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)

                    # Display Emotion + Confidence (Below Bbox)
                    cv2.putText(frame, text, (x1, y2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Annotated video saved as {output_video}")

def json_to_xlsx(json_path, excel_output, total_clusters, total_frames):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize DataFrame: rows = frames, cols = clusters
    df = pd.DataFrame(index=range(1, total_frames + 1),
                      columns=[f"Cluster {i}" for i in range(1, total_clusters + 1)])

    # Fill in emotion values
    for cluster_label, entries in data.items():
        # Extract numeric cluster index (e.g., "Cluster 15" → 15)
        cluster_id = int(cluster_label.split()[-1])
        
        # Convert to sequential index if needed (e.g., remap 15 → 1 if your lowest is 15)
        adjusted_cluster_id = cluster_id - min(int(c.split()[-1]) for c in data.keys()) + 1

        # Skip if adjusted cluster exceeds user-defined range
        if adjusted_cluster_id > total_clusters:
            continue

        for entry in entries:
            frame = entry['frame']
            emotion = entry['emotion']
            if frame <= total_frames:
                df.at[frame, f"Cluster {adjusted_cluster_id}"] = emotion

    # Export to Excel
    df.to_excel(excel_output, index_label='Frame')
    print(f"Saved to: {excel_output}")
    
def remove_files(file_paths):
    for path in file_paths:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except PermissionError:
            print(f"No permission to delete: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

# determine rejected IDs from individually annotated video 
excluded_ids = {"33"}

# computing tracking statistics
total_clusters, total_frames = compute_tracking_statistics(timestep_json, processed_json, IOU_THRESHOLD, fps)

# grouping and merging clusters
group_and_merge_clusters(processed_json, cluster_json)

# assigning clusters to detections
assign_clusters_to_detections(timestep_json, cluster_json, track_across_cluster)

# cluster-based annotation
annotate_video_cluster(original_video, track_across_cluster, cluster_output_video)

# conversion to excel
json_to_xlsx(track_across_cluster, excel_output, total_clusters, total_frames)

# clear files
files_to_delete = [
    timestep_json,
    cluster_json,
    #individual_output_video,
    track_across_cluster
]

remove_files(files_to_delete)