---
name: launch_website
description: >-
  Use this skill whenever the user says "website", asks to launch or open the website,
  start or verify the RxVLM Studio web server, or view the interactive medical OCR dashboard.
---

# Launch Website Skill

This skill automates launching, hosting, and verifying the **RxVLM Studio: Medical OCR & Knowledge Graph Pipeline** web application.

## Trigger Keyword
- **Keyword**: `website`
- **Aliases**: "launch website", "open website", "start server", "run web server", "view app"

## Operational Workflow

When this skill is triggered by the user saying "website" (or requesting to view/launch the web application), execute the following steps in sequence:

### Step 1: Check and Start the Flask Server
1. Check if the Flask server is already actively running on port 5000:
   - In PowerShell, run:
     ```powershell
     Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
     ```
   - Or test endpoint connectivity:
     ```powershell
     Invoke-RestMethod -Uri http://127.0.0.1:5000/api/prescriptions -ErrorAction SilentlyContinue
     ```
2. If the server is not running or not responding:
   - Run `python web_server.py` from the project directory (`C:\Users\ANOOP\Downloads\Extracted_DataDump\DataDump`) in the background as a daemon process:
     - Use `run_command` with `IsDaemon: true`, `CommandLine: "python web_server.py"`, and `Cwd: "c:\Users\ANOOP\Downloads\Extracted_DataDump\DataDump"`.
3. Allow 1-2 seconds for the server to bind to port 5000 and initialize routes.

### Step 2: Open Browser or Provide Clickable URL
1. Open the browser to the application:
   ```powershell
   Start-Process "http://127.0.0.1:5000"
   ```
2. Always provide a clear, clickable markdown link in the response:
   - 👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Step 3: Validate and Confirm Studio Components
1. Confirm the backend API is live and serving the complete dataset by checking `http://127.0.0.1:5000/api/prescriptions`:
   - Verify `total: 730` records are served.
   - Confirm partitions are available:
     - **Mapped Dataset** (27 canonical prescriptions with 1-to-1 images and JSON)
     - **Training Set** (602 samples)
     - **Testing Set** (100 benchmark samples)
2. Confirm the frontend UI components are active:
   - Three-panel responsive layout with resizable mouse-tracking splitters (`#sidebar-resizer` and `#assistant-resizer`).
   - Partition filter chips (`All (700+)`, `Mapped`, `Training (600)`, `Testing (100)`).
   - Structured medication table with full columns (`Drug Name`, `Form`, `Dosage`, `Frequency`, `Duration`, `Instructions`).
   - Rx Assistant AI sidebar with active contextual audit capability.
3. Present a concise confirmation message to the user confirming the status, dataset counts, and the live link.
