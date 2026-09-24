# Migrations (TRD 16.4 / 17.3)
The schema is defined in `backend/app/models/entities.py`. Generate the initial revision once on a machine with the
dependencies installed, review it, and commit it:

    pip install alembic && alembic init -t generic backend/migrations_env   # or copy env.py that imports Base
    alembic revision --autogenerate -m "initial schema"
    alembic upgrade head

CI already checks that the schema builds on a clean database. `scripts/seed_db.py` uses `create_all` for the demo only.
