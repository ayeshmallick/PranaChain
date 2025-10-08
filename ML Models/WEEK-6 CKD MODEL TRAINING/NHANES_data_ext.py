import requests
from io import BytesIO
import pandas as pd

# Let's actually SEE what the server is returning
url = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALB_CR_J.XPT"

print("Testing one specific URL to see what we get...")
print(f"URL: {url}\n")

response = requests.get(url, timeout=30, allow_redirects=True)

print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Content-Length: {len(response.content)} bytes")
print(f"\nFirst 500 characters of response:")
print(response.content[:500])
print(f"\n\nAs text (first 500 chars):")
try:
    print(response.text[:500])
except:
    print("Cannot decode as text")

# Try saving it to see the file
with open('test_download.xpt', 'wb') as f:
    f.write(response.content)

print("\n\nSaved response to 'test_download.xpt'")
print("Check if this file opens in a text editor to see what we actually got.")