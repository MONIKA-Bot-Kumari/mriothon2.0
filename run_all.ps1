Write-Host "Installing required Python packages..."
pip install fastapi uvicorn python-multipart --user

Write-Host "Starting backend server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python backend.py"

Start-Sleep -Seconds 2

Write-Host "Starting frontend server on http://localhost:5500 ..."
python -m http.server 5500
