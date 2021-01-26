import requests


url = 'https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16'
r = requests.get(url, allow_redirects=True)

open('VisualStudioInstaller.exe', 'wb').write(r.content)

os.system('')
