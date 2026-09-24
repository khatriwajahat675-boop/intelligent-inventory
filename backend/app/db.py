from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_Session = None


def init_engine(url: str | None = None):
    """Create the engine lazily so tests can inject sqlite."""
    global _engine, _Session
    _engine = create_engine(url or get_settings().database_url, pool_pre_ping=True, future=True)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


def get_db() -> Iterator[Session]:
    if _Session is None:
        init_engine()
    db = _Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()      # fail closed: no half-created PO/recommendation (EC-49)
        raise
    finally:
        db.close()


from contextlib import contextmanager


@contextmanager
def session_scope() -> Iterator[Session]:
    """Own transaction for background jobs (worker / BackgroundTasks)."""
    if _Session is None:
        init_engine()
    db = _Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
