import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_history.db")

conn = sqlite3.connect(DB_PATH)

# Add username column if it doesn't exist
try:
    conn.execute("ALTER TABLE transactions ADD COLUMN username TEXT")
    print("✅ Column 'username' added successfully!")
except Exception as e:
    print(f"ℹ️ {e}")

conn.commit()
conn.close()
print("✅ Database fixed!")
