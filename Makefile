# The hero video from the published results (lx/results/video_*.json, address_map*.json):
# build/video/out/{hero_16x9,hero_9x16,zoom_clip}.mp4, boot_loop.gif, thumbnails, qa.json.
# Needs the Python environment of the README's "Reproduce", scripts/setup.sh's models/ and ref/,
# ffmpeg, and Node >= 22.20 (NODE_BIN=/path/to/node/bin if the default node is older).
PYTHON ?= python

.PHONY: video
video:
	cd lx && PYTHONPATH=. $(PYTHON) video/render.py
