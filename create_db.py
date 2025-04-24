import sqlite3

conn = sqlite3.connect("sample.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS employees")

cursor.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INTEGER,
    hired_date TEXT
)
""")

cursor.executemany("""
INSERT INTO employees (name, department, salary, hired_date)
VALUES (?, ?, ?, ?)
""", [
    ("Alice", "Sales", 65000, "2021-03-12"),
    ("Bob", "Engineering", 85000, "2020-07-01"),
    ("Carol", "HR", 60000, "2019-08-20"),
    ("David", "Sales", 72000, "2022-01-15")
])

conn.commit()
conn.close()
print("✅ Database created successfully.")
