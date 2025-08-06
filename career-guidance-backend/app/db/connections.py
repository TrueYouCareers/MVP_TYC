# This module holds the global database connection instances initialized by app.main
# and provides an accessor function for routes and services.

mongodb_client = None
mysql_connection_instance = None


def get_db_connections():
    """
    Returns the initialized MySQL and MongoDB connections.
    Raises a RuntimeError if connections are not initialized.
    """
    if mysql_connection_instance is None or mongodb_client is None:
        # This should ideally not happen if the app lifespan manager works correctly.
        raise RuntimeError(
            "Database connections not initialized. Ensure the application lifespan manager has run.")
    return mysql_connection_instance, mongodb_client


def set_db_connections(mysql_conn, mongo_client):
    """
    Sets the global database connection instances.
    Called by the application's lifespan manager.
    """
    global mysql_connection_instance, mongodb_client
    mysql_connection_instance = mysql_conn
    mongodb_client = mongo_client


def clear_db_connections():
    """
    Clears the global database connection instances.
    Called by the application's lifespan manager on shutdown.
    """
    global mysql_connection_instance, mongodb_client
    mysql_connection_instance = None
    mongodb_client = None
