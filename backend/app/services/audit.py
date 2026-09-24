from sqlalchemy.orm import Session

from backend.app.models.entities import AuditEvent


def record(db: Session, actor: str, action: str, entity_type: str, entity_id, *, before=None, after=None,
           reason: str | None = None, policy_version: str | None = None, model_version: str | None = None,
           request_id: str | None = None) -> AuditEvent:
    """FR-018 / PRD 7.3 - written in the SAME transaction as the change it describes."""
    ev = AuditEvent(actor=actor, action=action, entity_type=entity_type, entity_id=str(entity_id),
                    before_json=before, after_json=after, reason=reason, policy_version=policy_version,
                    model_version=model_version, request_id=request_id)
    db.add(ev)
    return ev
