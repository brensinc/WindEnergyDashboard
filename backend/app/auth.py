import os
from fastapi import Header, HTTPException
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

async def get_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()

    # Validate token + fetch user
    res = supabase.auth.get_user(token)
    user = res.user
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user
