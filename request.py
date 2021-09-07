import requests

url = 'http://localhost:5000/results'

r = requests.post(url,json={'tweet':'This is a test tweet yay!'})

if r:
    print(r.json())
else:
    print("Nothing to return")