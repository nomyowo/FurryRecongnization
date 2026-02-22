import sqlite3

DB_NAME = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            type TEXT NOT NULL, -- 'lib' or 'query'
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # New table for multiple labels support
    conn.execute('''
        CREATE TABLE IF NOT EXISTS image_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
        )
    ''')

    # Check if migration is needed (if image_labels is empty but images has data)
    cursor = conn.execute("SELECT COUNT(*) FROM image_labels")
    if cursor.fetchone()[0] == 0:
        print("Migrating labels to image_labels table...")
        # Get all images with non-empty labels
        images = conn.execute("SELECT id, label FROM images WHERE label != '' AND label IS NOT NULL").fetchall()
        for img in images:
            # Split by comma if multiple labels exist in old format, or just take the single label
            labels = [l.strip() for l in img['label'].split(',') if l.strip()]
            for lbl in labels:
                conn.execute("INSERT INTO image_labels (image_id, label) VALUES (?, ?)", (img['id'], lbl))
        conn.commit()
        print("Migration complete.")

    conn.commit()
    conn.close()

def get_lib_labels_stats():
    conn = get_db_connection()
    try:
        db_labels = conn.execute('SELECT label, COUNT(*) as count FROM images WHERE type = ? GROUP BY label ORDER BY label', ('lib',)).fetchall()
        return [{"label": row['label'], "count": row['count']} for row in db_labels if row['label']]
    finally:
        conn.close()

def get_lib_images(label=None):
    conn = get_db_connection()
    try:
        if label:
            return conn.execute('SELECT * FROM images WHERE type = ? AND label = ? ORDER BY id DESC', ('lib', label)).fetchall()
        return conn.execute('SELECT * FROM images WHERE type = ? ORDER BY id DESC', ('lib',)).fetchall()
    finally:
        conn.close()

def add_image(filename, label, image_type, timestamp):
    conn = get_db_connection()
    try:
        cursor = conn.execute('INSERT INTO images (filename, label, type, timestamp) VALUES (?, ?, ?, ?)',
                     (filename, label, image_type, timestamp))
        image_id = cursor.lastrowid
        conn.commit()
        return image_id
    finally:
        conn.close()

def get_query_labels_stats():
    conn = get_db_connection()
    try:
        # Fetch distinct labels and counts from image_labels joined with images where type is query
        db_labels = conn.execute('''
            SELECT il.label, COUNT(DISTINCT il.image_id) as count 
            FROM image_labels il
            JOIN images i ON i.id = il.image_id 
            WHERE i.type = 'query' 
            GROUP BY il.label 
            ORDER BY il.label
        ''').fetchall()

        labels = []

        # Check for unlabeled query images
        unlabeled_count = conn.execute("SELECT COUNT(*) FROM images WHERE type = 'query' AND id NOT IN (SELECT image_id FROM image_labels)").fetchone()[0]
        if unlabeled_count > 0:
            labels.append({'label': '未检测到', 'original_label': '', 'count': unlabeled_count})

        for row in db_labels:
            labels.append({'label': row['label'], 'original_label': row['label'], 'count': row['count']})

        return labels
    finally:
        conn.close()

def get_query_images(label=None):
    conn = get_db_connection()
    try:
        images = []
        if label == "未检测到":
             # Special case: fetch images with no label relations
             images = conn.execute("SELECT * FROM images WHERE type = 'query' AND id NOT IN (SELECT image_id FROM image_labels) ORDER BY id DESC").fetchall()
        elif label:
             images = conn.execute('''
                SELECT i.* 
                FROM images i
                JOIN image_labels il ON i.id = il.image_id
                WHERE i.type = 'query' AND il.label = ? 
                ORDER BY i.id DESC
             ''', (label,)).fetchall()
        else:
             # If no label is specified (though usually handled by view logic), we might want all?
             # The controller logic separates folder view vs image view.
             # If this is called without label, maybe return empty or all?
             # Let's assume this returns all for now if needed, or caller handles it.
             pass

        # Attach concatenated labels to images for display
        images_with_labels = []
        for img in images:
            img_dict = dict(img)
            labels = conn.execute("SELECT label FROM image_labels WHERE image_id = ?", (img['id'],)).fetchall()
            img_dict['label'] = ", ".join([l[0] for l in labels])
            images_with_labels.append(img_dict)

        return images_with_labels
    finally:
        conn.close()

def add_image_labels(image_id, labels):
    conn = get_db_connection()
    try:
        for lbl in labels:
            conn.execute('INSERT INTO image_labels (image_id, label) VALUES (?, ?)', (image_id, lbl))
        conn.commit()
    finally:
        conn.close()

def get_image_type(image_id):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT type FROM images WHERE id = ?', (image_id,)).fetchone()
        return row['type'] if row else None
    finally:
        conn.close()

def update_image_label_legacy(image_id, label):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE images SET label = ? WHERE id = ?', (label, image_id))
        conn.commit()
    finally:
        conn.close()

def update_image_labels_query(image_id, normalized_label_str, label_list):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE images SET label = ? WHERE id = ?', (normalized_label_str, image_id))
        conn.execute('DELETE FROM image_labels WHERE image_id = ?', (image_id,))
        for lbl in label_list:
            conn.execute('INSERT INTO image_labels (image_id, label) VALUES (?, ?)', (image_id, lbl))
        conn.commit()
    finally:
        conn.close()

def get_all_query_images_with_labels():
    conn = get_db_connection()
    try:
        # Get all query images
        images = conn.execute("SELECT * FROM images WHERE type = ?", ('query',)).fetchall()
        result = []
        for img in images:
            # Get all labels for this image
            labels = conn.execute("SELECT label FROM image_labels WHERE image_id = ?", (img['id'],)).fetchall()
            label_list = [l[0] for l in labels]
            result.append({'image': img, 'labels': label_list})
        return result
    finally:
        conn.close()

def get_query_images_by_label_raw(label):
    conn = get_db_connection()
    try:
        if label == "未检测到":
            images = conn.execute("SELECT * FROM images WHERE type = ? AND id NOT IN (SELECT image_id FROM image_labels)", ('query',)).fetchall()
        else:
            images = conn.execute('''
                SELECT i.* 
                FROM images i
                JOIN image_labels il ON i.id = il.image_id
                WHERE i.type = 'query' AND il.label = ?
            ''', (label,)).fetchall()
        return images
    finally:
        conn.close()
