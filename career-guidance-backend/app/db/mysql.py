import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()


def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=os.getenv("MYSQL_PORT", "3306"),
            user=os.getenv("MYSQL_USER", "root"),
            # Consider changing default password source
            password=os.getenv("MYSQL_PASSWORD", "MySQL04112004_"),
            database=os.getenv("MYSQL_DATABASE", "llm_schema")
        )

        if connection.is_connected():
            print(
                f"MySQL connected to database: {os.getenv('MYSQL_DATABASE', 'llm_schema')}")
            return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None


def close_mysql_connection(connection):
    if connection and connection.is_connected():
        connection.close()
        print("MySQL connection closed")
