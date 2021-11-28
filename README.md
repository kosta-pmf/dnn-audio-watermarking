# dnn-audio-watermarking
DNN-based audio watermarking \
You can use our pretrained models on the samples we provided in the folder dataset. 
To start, restore the embedder and the detector models:
```
from evaluation import restore_model
embedder = restore_model("embedder_model")
detector = restore_model("detector_model")
```
To test embedding process call:
```
from main import embed
watermarked_signal = embed(embedder, signal)
```
To test detection call:
```
from main import detect
watermark = detect(detector, watermarked_signal)
```
In order to train your own model using our architecture you can use provided functions stored in python files like train etc.
