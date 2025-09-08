from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from cryptography.hazmat.primitives.asymmetric import rsa
from jose.utils import base64url_decode
import os, requests
from dotenv import load_dotenv

load_dotenv()
COGNITO_JWKS_URL = os.getenv("COGNITO_JWKS_URL")
COGNITO_AUDIENCE = os.getenv("COGNITO_CLIENT_ID")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_REGION = os.getenv("AWS_REGION")
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
COGNITO_GROUPS_CLAIM, COGNITO_ROLE_CLAIM = "cognito:groups", "custom:role"

router = APIRouter(prefix="/auth", tags=["Auth"])
_jwks_cache = None

def get_jwks():
    global _jwks_cache
    if _jwks_cache is None:
        _jwks_cache = requests.get(COGNITO_JWKS_URL).json()
    return _jwks_cache

def get_public_key(token: str):
    jwks = get_jwks()
    kid = jwt.get_unverified_header(token).get("kid")
    key = next((k for k in jwks["keys"] if k["kid"] == kid), None)
    if not key: raise JWTError("Public key not found.")
    n = int.from_bytes(base64url_decode(key["n"].encode()), "big")
    e = int.from_bytes(base64url_decode(key["e"].encode()), "big")
    return rsa.RSAPublicNumbers(e, n).public_key()

async def cognito_token_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path.startswith(("/uploads", "/webhook", "/ws", "/notifications")):
        return await call_next(request)
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})
    token = auth.split(" ")[1]
    try:
        payload = jwt.decode(
            token,
            get_public_key(token),
            algorithms=["RS256"],
            audience=COGNITO_AUDIENCE,
            issuer=COGNITO_ISSUER,
            options={"verify_at_hash": False}
        )
        request.state.cognito_user = payload
    except JWTError as e:
        return JSONResponse(status_code=401, content={"detail": f"Invalid token: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=401, content={"detail": f"Token validation error: {str(e)}"})
    return await call_next(request)
