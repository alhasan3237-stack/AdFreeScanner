# Test bench (optional, runs on a computer, not part of the app)
Python copies of the app's edge detection (v3.py) and filters (f3.py), with a generator of hard
synthetic document photos (synth.py). `python evaluate.py v3 60` prints the detection accuracy.
Needs: pip install opencv-python-headless==4.10.0.84 numpy pillow

## Retraining the page-corner network (docquad.onnx) on a GPU
`docquad/train_big.py` fine-tunes the app's own `app/src/main/assets/docquad.onnx` on freshly
generated scenes (`docquad/gen2.py`, a new scene for every sample, so any sample size fits on disk)
and exports the best epoch back into the same OpenCV-compatible graph (`docquad/onnxtorch.py`).
It scores the network before training and after each epoch on two held-out sets (unseen gen2
scenes, and `synth.py` photos from a different generator) and keeps the best.

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install opencv-python-headless==4.10.0.84 numpy pillow onnx
    set DQ_PHOTOS=<folder with OpenCV's samples/data images (backgrounds, covers, pictures)>
    python docquad/train_big.py ../app/src/main/assets/docquad.onnx <work folder> 60000 4
    python docquad/onnxtorch.py <work folder>/docquad_best.onnx   (checks it matches OpenCV)

Then copy `docquad_best.onnx` over `app/src/main/assets/docquad.onnx`. Fonts come from
`docquad/fonts.txt` if present, else from the computer's installed fonts.
