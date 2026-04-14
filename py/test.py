from db import get_client
client = get_client()

users = client.table("users").select("*", count="exact").execute()
activity = client.table("activity").select("*", count="exact").execute()
print("Users:", users.count)
print("Activity rows:", activity.count)