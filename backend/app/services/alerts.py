from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.entities import Alert


def raise_alert(db: Session, type_: str, severity: str, entity_type: str, entity_id, message: str,
                dedupe_key: str | None = None) -> Alert | None:
    """EC-38/FR-016: at most one OPEN alert per unresolved event."""
    key = dedupe_key or f"{type_}:{entity_type}:{entity_id}"
    if db.scalar(select(Alert).where(Alert.dedupe_key == key, Alert.status == "open")):
        return None
    a = Alert(type=type_, severity=severity, entity_type=entity_type, entity_id=str(entity_id),
              dedupe_key=key, message=message)
    db.add(a)
    return a
