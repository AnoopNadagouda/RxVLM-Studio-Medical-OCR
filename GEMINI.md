# Antigravity Workspace Guidelines for RxVLM Studio

## Trigger: "website"
Whenever the user's prompt is or contains the trigger keyword **"website"** (e.g., "website", "launch website", "open website", "show website"):
The agent MUST immediately execute the `launch_website` workflow:
1. **Start Server**: Check if `web_server.py` is actively running on `http://127.0.0.1:5000`. If not, start it in the background (`python web_server.py` in `C:\Users\ANOOP\Downloads\Extracted_DataDump\DataDump`).
2. **Open Browser & Provide Link**: Launch the browser (`Start-Process "http://127.0.0.1:5000"`) and provide a clickable markdown link: [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. **Confirm Studio Health**: Confirm that the studio is live, serving all 730 records across dataset partitions (Mapped, Training, Testing), with the resizable three-panel layout and active Rx Assistant AI.
