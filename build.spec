# PyInstaller spec for LiveCaptions.exe (onedir: ~500MB folder with CUDA libs).
# Build:  pyinstaller build.spec
# Output: dist/LiveCaptions/LiveCaptions.exe
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

import os, glob
binaries = []
datas = []
hiddenimports = []

# CTranslate2 needs system CUDA DLLs (cuBLAS, cuDNN, CUDA runtime). They ship with
# torch's wheel, but we don't want the 1+ GB torch_cuda.dll / torch_cpu.dll that come with it.
# Pull ONLY the cuBLAS / cuDNN / cudart DLLs (ctranslate2's actual dependency set).
_TORCH_LIB = r"C:\Users\User\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
if os.path.isdir(_TORCH_LIB):
    for pat in ("cublas*.dll", "cublasLt*.dll", "cudart*.dll", "cudnn*.dll"):
        for f in glob.glob(os.path.join(_TORCH_LIB, pat)):
            binaries.append((f, "."))

# CTranslate2 bundles its own core DLLs.
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("pyaudiowpatch")

# faster_whisper has a few data bits (tokenizers/model config files ship via HF cache at runtime;
# only code+tiktoken regex files need bundling).
datas += collect_data_files("faster_whisper")
datas += collect_data_files("tokenizers")

hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += ["pyaudiowpatch", "tokenizers", "huggingface_hub"]

a = Analysis(
    ["captions.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=[
        "matplotlib", "pandas", "scipy", "PIL", "tests", "test",
        # faster-whisper/ctranslate2 do NOT need torch at runtime — strip ~3 GB of torch+CUDA DLLs.
        "torch", "torchvision", "torchaudio",
        "sympy", "networkx",  # torch-only deps
        "sentencepiece",  # not used for .en models
        # NOTE: do NOT exclude filelock or jinja2 — huggingface_hub needs them
        # for download locking when a new model is fetched.
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveCaptions",
    console=False,              # pythonw-equivalent (no console window)
    disable_windowed_traceback=False,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LiveCaptions",
)
