import sqlite3
import numpy as np
import cv2
import os
import random

# --- CONFIG ---
DB_PATH = "outputs/test.db"
IMAGES_DIR = "data/train/ETS"  # Update this if your images are elsewhere
OUTPUT_IMG = "match_debug.png"
# --------------

def pair_id_to_image_ids(pair_id):
    # COLMAP standard formula: pair_id = image_id1 * 2147483647 + image_id2
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

def visualize_random_match():
    if not os.path.exists(DB_PATH):
        print(f"❌ DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Get matches (Using only columns that actually exist)
    print("Reading matches from DB...")
    cursor.execute("SELECT pair_id, rows, data FROM matches WHERE rows > 15")
    matches = cursor.fetchall()
    
    if not matches:
        print("❌ No matches with >15 inliers found.")
        conn.close()
        return

    # 2. Pick a random pair
    pair = random.choice(matches)
    pair_id, num_matches, match_blob = pair
    
    # 3. Decode the IDs
    id1, id2 = pair_id_to_image_ids(pair_id)
    print(f"Selected Pair ID: {pair_id} -> Image IDs: {id1} & {id2}")
    
    # 4. Get Image Names
    cursor.execute("SELECT name FROM images WHERE image_id = ?", (id1,))
    res1 = cursor.fetchone()
    cursor.execute("SELECT name FROM images WHERE image_id = ?", (id2,))
    res2 = cursor.fetchone()

    if not res1 or not res2:
        print(f"❌ Could not find image names for IDs {id1} or {id2}")
        conn.close()
        return
        
    name1, name2 = res1[0], res2[0]
    
    # 5. Get Keypoints
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (id1,))
    kpts1_blob = cursor.fetchone()[0]
    kpts1 = np.frombuffer(kpts1_blob, dtype=np.float32).reshape(-1, 2)
    
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (id2,))
    kpts2_blob = cursor.fetchone()[0]
    kpts2 = np.frombuffer(kpts2_blob, dtype=np.float32).reshape(-1, 2)
    
    # 6. Decode Match Indices
    match_indices = np.frombuffer(match_blob, dtype=np.uint32).reshape(-1, 2)

    conn.close()

    # 7. Visualization
    path1 = os.path.join(IMAGES_DIR, name1)
    path2 = os.path.join(IMAGES_DIR, name2)
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"❌ Image file missing:\n   {path1}\n   {path2}")
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # OpenCV DrawMatches
    cv_kpts1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts1]
    cv_kpts2 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts2]
    
    dmatches = []
    for i in range(len(match_indices)):
        # COLMAP stores matches as (idx_in_image1, idx_in_image2)
        dmatches.append(cv2.DMatch(_queryIdx=match_indices[i,0], _trainIdx=match_indices[i,1], _distance=0))

    out_img = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, dmatches, None, flags=2)
    
    cv2.imwrite(OUTPUT_IMG, out_img)
    print(f"✅ Success! Saved match visualization to: {OUTPUT_IMG}")
    print(f"   Pair: {name1} <-> {name2} ({num_matches} inliers)")

if __name__ == "__main__":
    visualize_random_match()