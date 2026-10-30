# Setting up the scouting PC (Windows or Linux, RTX 3060)

1. **Install the prerequisites:**
   - Python 3.11 or 3.12
   - Git
   - FFmpeg, which must be on PATH: on Windows `winget install ffmpeg`, on Linux `apt install ffmpeg`
   - Node.js, which yt-dlp uses to solve YouTube's JavaScript challenges: `winget install OpenJS.NodeJS`
2. **Install PyTorch with CUDA first,** then the project:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -e .
   python -c "import torch; print(torch.cuda.is_available())"   # must print True
   ```
3. **Optional, recommended: TensorRT** for about 2–3× faster detection: `pip install tensorrt`. `autoscout train-robots`
   then exports `autoscout-robots.engine` automatically, and the pipeline prefers it.
4. **Secrets.** Create `.secrets/.env`:
   ```
   TBA_API_KEY=your_key
   YT_COOKIES=.secrets/cookies.txt
   ```
   Cookies are optional on a home connection. They help when YouTube asks to "confirm you're not a bot".
5. **Copy `data/models/`** from the build machine, or retrain with `autoscout train-robots`.
6. **Run the dashboard:** `autoscout serve --host 0.0.0.0` lets the rest of the team open it on the local network.

**Throughput.** Robot detection at 10 Hz on 1080p uses about 20% of the 3060, and decoding plus launch tracking uses
2–3 CPU cores. To process several matches in parallel, run `autoscout run-event` in 2–3 terminals with different
`--levels` or events.
