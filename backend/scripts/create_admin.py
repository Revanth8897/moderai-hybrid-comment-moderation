
import getpass
import sys

from sqlalchemy import select

from backend.app.database.models import User
from backend.app.database.session import SessionLocal
from backend.app.security.password import hash_password


def create_admin():
    db = SessionLocal()

    try:
        # Allow bootstrap only when no ADMIN account exists.
        existing_admin = db.scalar(
            select(User).where(User.role == "ADMIN")
        )

        if existing_admin:
            print("An ADMIN account already exists. Bootstrap stopped.")
            return

        email = input("Enter admin email: ").strip().lower()

        if not email or "@" not in email:
            print("Invalid email address.")
            return

        existing_user = db.scalar(
            select(User).where(User.email == email)
        )

        if existing_user:
            print("This email is already registered.")
            return

        password = getpass.getpass("Enter admin password: ")
        confirm_password = getpass.getpass("Confirm admin password: ")

        if len(password) < 12:
            print("Password must contain at least 12 characters.")
            return

        if password != confirm_password:
            print("Passwords do not match.")
            return

        admin = User(
            email=email,
            hashed_password=hash_password(password),
            role="ADMIN",
            is_active=True,
        )

        db.add(admin)
        db.commit()
        db.refresh(admin)

        print("\nADMIN account created successfully.")
        print(f"Admin ID: {admin.id}")
        print(f"Admin email: {admin.email}")
        print(f"Admin role: {admin.role}")

    except Exception:
        db.rollback()
        print("Admin creation failed. Database changes were rolled back.")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    create_admin()
