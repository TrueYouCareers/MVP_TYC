from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig
from alembic import context
from app.db.session import Base
from app.models import user, profile  # Import your models here
import os  # Added import


# Interpret the config file for Python logging.
# This line sets up loggers, handlers, and formatters from the config file.
fileConfig(context.config.config_file_name)

# Add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


# Function to get database URL from environment variables
def get_url():
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "yoursecurepassword")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    db = os.getenv("MYSQL_DATABASE", "llm_schema")
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}"


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_url()  # Use the function to get URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    # Get the Alembic config object
    config = context.config

    # Set the sqlalchemy.url in the config object using environment variables
    config.set_main_option("sqlalchemy.url", get_url())

    connectable = engine_from_config(
        # Use the modified config
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Enable type comparison for autogenerate
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
