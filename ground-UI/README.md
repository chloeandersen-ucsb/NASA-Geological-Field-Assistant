# ground-UI

Browser-based mission review tool for mission control. Lets you load a `sage_ground_data` folder (transferred from the Jetson) and browse rocks, voice notes, and images by mission/sample.

Static HTML/JS page opened locally.

## setup

1. Edit `config.json` to point at your local `sage_ground_data` folder:
```json
{ "sage_ground_data_path": "C:\\Users\\You\\capstone\\sage_ground_data" }
```

2. Open `index.html` in a browser (or serve it locally if CORS is an issue)

## generating missions.json

If you have raw `sage_ground_data` folder structure but no `missions.json`, regenerate it:

```bash
python parse_mission_data.py <path_to_sage_ground_data>
```

This scans the folder for `rocks.jsonl`, `voice_notes.jsonl`, and timestamped images, then writes `missions.json`.

## what's in the UI

- **mission selector** - dropdown of all logged missions
- **sample tree** — rocks collected per mission, with ML classification + confidence
- **rock detail** — image, primary classification, feature breakdown, voice note transcript
- you can also drag-and-drop or file-import the transferred folder directly in the UI