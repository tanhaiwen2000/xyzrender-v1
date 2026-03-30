import urllib.request
import json

data = json.dumps({
    "file": "example.xyz", # Need a valid file
    "highlights": [{"atoms": "C", "color": "red"}]
}).encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:3000/api/render', data=data, headers={'Content-Type': 'application/json'})
try:
    response = urllib.request.urlopen(req)
    print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
