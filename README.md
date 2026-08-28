# facTruth

Flask fact-checking UI and API using SerpApi multi-engine search, Gemini evaluation, and an optional Twilio WhatsApp webhook.

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Set API keys in the current PowerShell session (do not put real keys in source files):

   ```powershell
   $env:SERPAPI_KEY = "your-serpapi-key"
   $env:GEMINI_API_KEY = "your-gemini-key"
   ```

3. Start the app:

   ```powershell
   python app.py
   ```

Open `http://127.0.0.1:5000`. Without keys, the app still starts and `/api/verify` returns a clear configuration message.

## Security note

Any credentials that were previously committed, pasted into code, or shared in uploads should be revoked/rotated in the provider dashboards. Keep new credentials in environment variables and out of version control.
