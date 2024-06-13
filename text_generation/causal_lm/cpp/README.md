# Text generation C++ samples  




### LLM

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.


> [!NOTE]
>Models should belong to the same family and have same tokenizers.

## Install OpenVINO

Install [OpenVINO Archives >= 2024.1](docs.openvino.ai/install). `master` and possibly the latest `releases/*` branch correspond to not yet released OpenVINO versions. https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/ can be used for these branches early testing. `<INSTALL_DIR>` below refers to the extraction location.

### Windows

```bat
git submodule update --init
<INSTALL_DIR>\setupvars.bat
cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j
```

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

#### Windows

```bat
<INSTALL_DIR>\setupvars.bat
python -m pip install --upgrade-strategy eager -r requirements.txt
REM Update openvino_tokenizers from the submodule
python -m pip install .\..\..\..\thirdparty\openvino_tokenizers\[transformers]
optimum-cli export openvino --trust-remote-code --weight-format fp16 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Run

### Usage:
`llm <MODEL_DIR> "<PROMPT>"`

### Examples:

#### Windows:
`.\build\Release\llm .\TinyLlama-1.1B-Chat-v1.0\ "Why is the Sun yellow?"`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
