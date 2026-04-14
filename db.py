from supabase import create_client, Client
import credentials

def get_client() -> Client:
    """Return an authenticated Supabase client."""
    return create_client(credentials.SUPABASE_URL, credentials.SUPABASE_KEY)

if __name__ == "__main__":
    client = get_client()
    print("Supabase client connected:", client.table("users").select("count", count="exact").execute())