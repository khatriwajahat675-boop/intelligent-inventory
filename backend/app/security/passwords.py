"""bcrypt hashing (TRD 15.1). Plaintext passwords are never stored or logged."""
import bcrypt


def hash_password(plain: str) -> str:
    if len(plain) < 10:
        raise ValueError("password must be at least 10 characters")
    return bcrypt.hashpw(plain.encode()[:72], bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode()[:72], hashed.encode())
    except ValueError:
        return False
