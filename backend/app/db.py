import os
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Service role lets backend bypass RLS IF you want;
# but we will still always filter by user_id.
supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
