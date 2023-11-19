import os
os.environ.setdefault("TQDM_DISABLE", "1") # HACK: Downloading a model has its own progress bar, loading it locally does not. There is no way to tell them apart ¯\_(ツ)_/¯

from .main import main

main()
