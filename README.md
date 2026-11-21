# Amed — Hospital Unit Twin (Gemini 4 NYC Hackathon)

Built from scratch during the Gemini 4 NYC Hackathon (Nov 21, 2026) for **Element 119**.

> "Nurses spend part of every shift walking supplies around the unit. Give us a floor plan, and we show what changes when robots take those trips."

## Architecture & Model Jobs
- **Gemini 4 Neon**: Extracts rooms (`label`, `type`, `polygon`) and doors (`position`, `rooms_connected`) from a floor plan image into `unit.json`.
- **Decisions API**: Verifies every extracted room and door against its cropped image region with calibrated confidence; re-extracts or drops low-confidence items into `verified_unit.json`.
- **Gemini 4 Hydrogen**: Parses spoken or typed shift run descriptions into a structured schedule (`run_type`, `origin`, `destination`, `frequency_minutes`).
- **Route Graph & SimPy**: Converts `verified_unit.json` to a NetworkX graph (meters) and simulates nurse-only vs. courier-robot shifts.
- **HTML5 Canvas Twin**: Animates nurse and robot paths side-by-side directly over the floor plan image, displaying minutes walked saved and stated assumptions.
