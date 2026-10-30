# Running on a Mac (Apple silicon: M1/M2/M3)

Robot detection and training run on the Apple GPU (PyTorch "MPS"); this is picked automatically.
Bumper-number OCR runs on the CPU (EasyOCR's GPU path is NVIDIA-only), which is still fast on M-series chips.

## 1. Install tools (Terminal)
```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 ffmpeg node git
```
`node` is used by yt-dlp to download YouTube videos.

## 2. Get the code and install
```bash
git clone https://github.com/ryanabraham1/frc-auto-scout.git
cd frc-auto-scout
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python -c "import torch; print(torch.backends.mps.is_available())"   # should print True
python -c "from autoscout import config; print(config.device())"     # should print mps
```

## 3. Secrets
Create `.secrets/.env` (this folder is git-ignored):
```
TBA_API_KEY=your_key_from_thebluealliance.com/account
```
Cookies are usually not needed from a home network. If YouTube says "confirm you're not a bot",
export cookies (see README) to `.secrets/cookies.txt` and add `YT_COOKIES=.secrets/cookies.txt`.

## 4. Run
```bash
autoscout serve                                   # dashboard: http://127.0.0.1:8000 (Processing tab shows progress bars)
autoscout run-match 2026cc_sf11m1 --debug-video   # one match (downloads the video itself)
autoscout run-event 2026cc                        # every match with a video
python scripts/render_demo.py 2026cc_sf11m1 data/videos/IOxILF_5-2U.mp4 demo/2026cc_sf11m1.mp4   # annotated video
```
Events that are already calibrated: 2026cc, 2026dal, 2026cosp, 2026txhou, 2025casj. Others need the
one-time Calibrate step (`autoscout calibrate <match_key>`, then the dashboard's Calibrate page).

For 2025 matches with reef close-up cameras, `autoscout aux <match_key> <video>` (already part of
`run-match` when the event config has `aux_views`).

## Notes
- Force a device with `AUTOSCOUT_DEVICE=cpu` (or `mps`) if anything misbehaves on the GPU.
- If an op is not supported on MPS, `export PYTORCH_ENABLE_MPS_FALLBACK=1` lets PyTorch fall back to CPU for it.
- Retrain the robot detector on the Mac GPU: `autoscout train-robots` (labels from the dashboard's Label page).
