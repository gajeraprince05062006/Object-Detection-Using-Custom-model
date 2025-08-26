import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="smart"
        )
        if conn.is_connected():
            return conn
        else:
            raise Exception("Failed to connect to database")
    except Error as e:
        print(f"[DB ERROR] {e}")
        raise  # Don't return None — let Flask show the real error
