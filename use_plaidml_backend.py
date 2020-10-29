import os

os.environ["RUNFILES_DIR"] = '/usr/local/share/plaidml'
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["PLAIDML_NATIVE_PATH"] = "/usr/local/lib/libplaidml.dylib"

import plaidml.keras

plaidml.keras.install_backend()
