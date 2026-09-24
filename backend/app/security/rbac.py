"""Server-side RBAC (PRD section 3, FR-002). Never rely on hidden UI buttons (TRD 15.1)."""
from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    MANAGER = "inventory_manager"
    OPERATOR = "operations_user"
    ANALYST = "analyst"
    SERVICE = "automation_service"


READ = {"products:read", "suppliers:read", "inventory:read", "forecasts:read", "recommendations:read",
        "purchase_orders:read", "alerts:read"}

PERMISSIONS: dict[Role, set[str]] = {
    Role.ANALYST: READ | {"audit:read", "export"},
    Role.OPERATOR: READ | {"sales:write", "inventory:adjust", "purchase_orders:receive"},
    Role.MANAGER: READ | {"products:write", "suppliers:write", "sales:write", "sales:import", "inventory:adjust",
                          "forecasts:run", "recommendations:evaluate", "recommendations:decide",
                          "purchase_orders:write", "purchase_orders:receive", "audit:read", "export", "config:read"},
    Role.SERVICE: READ | {"forecasts:run", "recommendations:evaluate", "recommendations:auto_approve",
                          "purchase_orders:write"},
    Role.ADMIN: set(),   # filled below: everything
}
PERMISSIONS[Role.ADMIN] = set().union(*PERMISSIONS.values()) | {"users:write", "config:write"}


def has_permission(role: str, permission: str) -> bool:
    try:
        return permission in PERMISSIONS[Role(role)]
    except ValueError:
        return False
