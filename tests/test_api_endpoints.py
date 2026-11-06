
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database.base import Base
from backend.app.database.session import get_db
from backend.app.services.comment_service import CommentService
from backend.app.database.models import User
from backend.app.security.password import verify_password
from backend.app.security.jwt import verify_access_token

from backend.app.api.routes import (
    comments as comments_routes,
    reviews as reviews_routes,
)

from moderation.comment_pipeline import CommentModerationPipeline
from moderation.review.manager import ReviewManager


@pytest.fixture
def client(monkeypatch):

    # 1. Create an isolated SQLite database.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=test_engine)

    TestSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )
    app.state.test_session_local = TestSessionLocal

    # 2. Override FastAPI database dependency.
    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # 3. Inject SQLite into the ReviewManager.
    test_review_manager = ReviewManager(
        session_factory=TestSessionLocal
    )

    # 4. Create a test pipeline and shared service.
    test_pipeline = CommentModerationPipeline(
        review_manager=test_review_manager
    )

    test_service = CommentService(
        pipeline=test_pipeline
    )

    # 5. Inject the same service into both API routes.
    monkeypatch.setattr(
        comments_routes,
        "comment_service",
        test_service,
    )

    monkeypatch.setattr(
        reviews_routes,
        "comment_service",
        test_service,
    )

    # 6. Create the HTTP test client.
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # 7. Remove the dependency override.
        app.dependency_overrides.pop(get_db, None)

        # 8. Clean up the test database.
        if hasattr(app.state, "test_session_local"):
            delattr(app.state, "test_session_local")
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


def create_test_review(client):

    response = client.post(
        "/comments/analyze",
        json={
            "comment_text": (
                "I really love this product. "
                "Thank you for building it!"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["error"] is None

    review = data["data"]["review"]

    assert review is not None

    return review



def create_authenticated_reviewer(client, email="reviewer@example.com"):
    password = "StrongPassword123!"

    registration = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert registration.status_code == 201

    user_id = registration.json()["data"]["id"]

    login = client.post(
        "/auth/login",
        data={
            "username": email,
            "password": password,
        },
    )

    assert login.status_code == 200

    token = login.json()["access_token"]

    return {
        "user_id": user_id,
        "headers": {
            "Authorization": f"Bearer {token}",
        },
    }

# --------------------------------------------------
# Application tests
# --------------------------------------------------

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert (
        response.json()["application"]
        == "HYBRID COMMENT MODERATION SYSTEM"
    )


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


# --------------------------------------------------
# Comment analysis tests
# --------------------------------------------------

def test_comment_analysis_creates_review(client):

    review = create_test_review(client)

    assert review["review_id"] is not None
    assert review["review_status"] == "PENDING"


def test_empty_comment_is_rejected(client):

    response = client.post(
        "/comments/analyze",
        json={"comment_text": ""},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"



# --------------------------------------------------
# Review retrieval tests
# --------------------------------------------------

def test_pending_reviews_endpoint(client):
    auth = create_authenticated_reviewer(
        client,
        "pending@example.com",
    )

    review = create_test_review(client)

    response = client.get(
        "/reviews/pending",
        headers=auth["headers"],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    result = data["data"]

    assert result["total"] >= 1
    assert result["has_more"] is False

    returned_review_ids = [
        item["review_id"]
        for item in result["reviews"]
    ]

    assert review["review_id"] in returned_review_ids


def test_get_individual_review(client):
    auth = create_authenticated_reviewer(
        client,
        "individual@example.com",
    )

    review = create_test_review(client)
    review_id = review["review_id"]

    response = client.get(
        f"/reviews/{review_id}",
        headers=auth["headers"],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["review_id"] == review_id


def test_unknown_review_returns_404(client):
    auth = create_authenticated_reviewer(
        client,
        "unknownreview@example.com",
    )

    response = client.get(
        "/reviews/unknown-review-id",
        headers=auth["headers"],
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"


# --------------------------------------------------
# Review workflow tests
# --------------------------------------------------

def test_complete_review_workflow(client):
    auth = create_authenticated_reviewer(
        client,
        "workflow@example.com",
    )

    review = create_test_review(client)
    review_id = review["review_id"]

    # Start the review.
    start_response = client.post(
        f"/reviews/{review_id}/start",
        json={},
        headers=auth["headers"],
    )

    assert start_response.status_code == 200

    start_data = start_response.json()

    assert start_data["success"] is True
    assert start_data["data"]["review_status"] == "IN_REVIEW"

    # Reviewer identity must come from the authenticated user.
    assert start_data["data"]["reviewer_id"] == auth["user_id"]

    # Submit the approval decision.
    decision_response = client.post(
        f"/reviews/{review_id}/decision",
        json={
            "decision": "APPROVE_FOR_DRAFT",
            "reason": "Comment is safe and deserves an answer.",
        },
        headers=auth["headers"],
    )

    assert decision_response.status_code == 200

    decision_data = decision_response.json()

    assert decision_data["success"] is True
    assert decision_data["data"]["review_status"] == "APPROVED"
    assert decision_data["data"]["review_decision"] == "APPROVE_FOR_DRAFT"

    # Approval does not authorize external actions.
    assert decision_data["data"]["automatic_action_allowed"] is False


def test_invalid_review_decision_returns_422(client):
    auth = create_authenticated_reviewer(
        client,
        "invaliddecision@example.com",
    )

    review = create_test_review(client)
    review_id = review["review_id"]

    # Start the review first.
    start_response = client.post(
        f"/reviews/{review_id}/start",
        json={},
        headers=auth["headers"],
    )

    assert start_response.status_code == 200

    # Submit an invalid decision.
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={
            "decision": "UNKNOWN",
            "reason": "Invalid decision test.",
        },
        headers=auth["headers"],
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


# --------------------------------------------------
# Review authentication tests
# --------------------------------------------------

@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/reviews/pending", None),
        ("GET", "/reviews/unknown-review-id", None),
        ("POST", "/reviews/test-review-id/start", {}),
        (
            "POST",
            "/reviews/test-review-id/decision",
            {"decision": "NO_RESPONSE"},
        ),
    ],
)
def test_review_endpoints_require_authentication(
    client,
    method,
    path,
    payload,
):
    response = client.request(
        method,
        path,
        json=payload,
    )

    assert response.status_code == 401


# --------------------------------------------------
# Authentication and registration tests
# --------------------------------------------------

def test_user_registration_success_and_password_hash(client):

    password = "StrongPassword123!"

    response = client.post(
        "/auth/register",
        json={
            "email": "reviewer@example.com",
            "password": password,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["success"] is True
    assert data["data"]["email"] == "reviewer@example.com"
    assert data["data"]["role"] == "REVIEWER"
    assert data["data"]["is_active"] is True

    # Password hash must never be exposed in the API response.
    assert "hashed_password" not in data["data"]
    assert "password" not in data["data"]

    # Verify the password is securely hashed in SQLite.
    TestSessionLocal = app.state.test_session_local

    with TestSessionLocal() as db:
        user = db.query(User).filter(
            User.email == "reviewer@example.com"
        ).one()

        assert user.hashed_password != password
        assert verify_password(password, user.hashed_password)


def test_duplicate_email_registration_returns_409(client):

    payload = {
        "email": "duplicate@example.com",
        "password": "StrongPassword123!",
    }

    first_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert first_response.status_code == 201

    # Same email with different capitalization.
    second_response = client.post(
        "/auth/register",
        json={
            "email": "Duplicate@Example.com",
            "password": "AnotherPassword123!",
        },
    )

    assert second_response.status_code == 409


def test_invalid_email_and_weak_password_are_rejected(client):

    invalid_email_response = client.post(
        "/auth/register",
        json={
            "email": "invalid-email",
            "password": "StrongPassword123!",
        },
    )

    assert invalid_email_response.status_code == 422

    weak_password_response = client.post(
        "/auth/register",
        json={
            "email": "weak@example.com",
            "password": "short",
        },
    )

    assert weak_password_response.status_code == 422


def test_public_registration_cannot_create_admin(client):

    response = client.post(
        "/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "StrongPassword123!",
            "role": "ADMIN",
        },
    )

    assert response.status_code == 201

    data = response.json()

    # Public registration must always create REVIEWER accounts.
    assert data["data"]["role"] == "REVIEWER"


def test_registration_does_not_expose_password_fields(client):

    response = client.post(
        "/auth/register",
        json={
            "email": "secure@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    response_data = response.json()["data"]

    assert "password" not in response_data
    assert "hashed_password" not in response_data



def test_user_login_success_returns_jwt_token(client):
    password = "StrongPassword123!"
    email = "login@example.com"

    # Register the user first.
    registration_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert registration_response.status_code == 201

    # Login using OAuth2 form fields.
    response = client.post(
        "/auth/login",
        data={
            "username": email,
            "password": password,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    
    
    # verify the jwt payload
    payload = verify_access_token(data["access_token"])
    
    assert payload["role"] == "REVIEWER"
    assert payload["sub"] == registration_response.json()["data"]["id"]


def test_user_login_with_wrong_password_returns_401(client):
    email = "wrongpassword@example.com"

    registration_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "StrongPassword123!",
        },
    )

    assert registration_response.status_code == 201

    response = client.post(
        "/auth/login",
        data={
            "username": email,
            "password": "IncorrectPassword123!",
        },
    )

    assert response.status_code == 401


def test_user_login_with_unknown_email_returns_401(client):
    response = client.post(
        "/auth/login",
        data={
            "username": "unknown@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_inactive_user_cannot_login(client):
    email = "inactive@example.com"
    password = "StrongPassword123!"

    registration_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert registration_response.status_code == 201

    # Deactivate the registered account in the test database.
    TestSessionLocal = app.state.test_session_local

    with TestSessionLocal() as db:
        user = (
            db.query(User)
            .filter(User.email == email)
            .one()
        )

        user.is_active = False
        db.commit()

    response = client.post(
        "/auth/login",
        data={
            "username": email,
            "password": password,
        },
    )

    assert response.status_code == 401



def test_get_current_user_profile_with_valid_token(client):
    email = "profile@example.com"
    password = "StrongPassword123!"

    # Register user.
    registration = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )
    assert registration.status_code == 201

    # Login and obtain JWT.
    login = client.post(
        "/auth/login",
        data={
            "username": email,
            "password": password,
        },
    )
    assert login.status_code == 200

    token = login.json()["access_token"]

    # Access protected profile endpoint.
    response = client.get(
        "/auth/me",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["email"] == email
    assert data["data"]["role"] == "REVIEWER"
    assert data["data"]["is_active"] is True

    # Sensitive password information must not be exposed.
    assert "password" not in data["data"]
    assert "hashed_password" not in data["data"]

# --------------------------------------------------
# Authentication invalidation tests
# --------------------------------------------------

def test_deactivated_user_token_is_rejected(client):
    auth = create_authenticated_reviewer(
        client,
        "deactivated@example.com",
    )

    # Deactivate the user directly in the test database.
    db = app.state.test_session_local()
    try:
        user = db.get(User, auth["user_id"])
        assert user is not None

        user.is_active = False
        db.commit()
    finally:
        db.close()

    # The previously issued token must no longer work.
    response = client.get(
        "/auth/me",
        headers=auth["headers"],
    )

    assert response.status_code == 401
    assert response.json()["success"] is False


def test_changed_user_role_invalidates_old_token(client):
    auth = create_authenticated_reviewer(
        client,
        "rolechange@example.com",
    )

    # Change the database role after the token was issued.
    db = app.state.test_session_local()
    try:
        user = db.get(User, auth["user_id"])
        assert user is not None

        user.role = "ADMIN"
        db.commit()
    finally:
        db.close()

    # The old token still contains REVIEWER, so it must be rejected.
    response = client.get(
        "/auth/me",
        headers=auth["headers"],
    )

    assert response.status_code == 401
    assert response.json()["success"] is False

def test_get_current_user_profile_without_token_returns_401(client):
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_get_current_user_profile_with_invalid_token_returns_401(client):
    response = client.get(
        "/auth/me",
        headers={
            "Authorization": "Bearer invalid.jwt.token",
        },
    )

    assert response.status_code == 401
