
from fastapi import Depends, HTTPException, status

from backend.app.database.models import User
from backend.app.dependencies.auth import get_current_user


VALID_ROLES = {"ADMIN", "REVIEWER"}


def require_roles(*allowed_roles: str):
    """Create a dependency that restricts access by user role."""

    if not allowed_roles:
        raise ValueError("At least one allowed role is required.")

    if not set(allowed_roles).issubset(VALID_ROLES):
        raise ValueError("Invalid role specified.")

    def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:

        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource.",
            )

        return current_user

    return role_checker
