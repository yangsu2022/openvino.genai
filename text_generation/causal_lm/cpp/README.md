# Text generation C++ samples  




### LLM

The program loads a tokenizer, a detokenizer and a model (`.xml` and `.bin`) to OpenVINO. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion. The pipeline is with model caching by default. The logits reducing optimization improves the first token latency.

### Convert Tokenizers
This pure C++ LLM pipeline has C++ implementation of openvino tokenizer to avoid the Python dependencies.
To convert tokenizer into OV model IR, create a python env with conda(or venv).
#### Windows

```bat
conda create -n convert_ov_tokenizer_model python=3.10
conda activate convert_ov_tokenizer_model
<INSTALL_DIR>\setupvars.bat
pip install openvino-tokenizers transformers>=4.40.2 torch>=2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu
python export_ov_tokenizer.py -m .\{YOUR_RELATIVE_PATH_OV_INT4} -o ov_tokenizer_models
conda deactivate
```
Notice:
- This script will generate model IR of `openvino_tokenizer` and `openvino_detokenizer` in the ov_tokenizer_models folder.
- Copy the OV IR(both xml and bin) into the same folder of your LLM IR, which is converted with native OpenVINO API instead of optimum-intel.

## Install OpenVINO and build LLM C++ pipeline

Download [2024.2.0rc2](https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.2.0rc2/windows/), if the 2024.2 release is still not available in [OpenVINO™ archives*](https://storage.openvinotoolkit.org/repositories/openvino/packages/). This OV built package is for C++ OpenVINO pipeline, no need to build the source code.

Notice: 
- Extract the zip file in any location and set the environment variables with dragging this `setupvars.bat` in the terminal `Command Prompt`.
- `setupvars.ps1` is used for terminal `PowerShell`.
- `<INSTALL_DIR>` below refers to the extraction location.

### Windows
Install latest VS2022, and run the following CMD in the terminal `Command Prompt`.

```bat
git submodule update --init
<INSTALL_DIR>\setupvars.bat
cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j8
```
Notice:
- The ov tokenizer in the third party needs several minutes to build. Set 8 for -j option to specify the number of parallel jobs. 
- Once the cmake finishes, check the llm.exe file in the relative path `.\build\Release\llm.exe`. 
- If Cmake completed without errors, but not find exe, please open the `.\build\llm.sln` in VS2022, and set the solution configuration as Release instead of Debug, Then build the llm project within VS2022 again.

  
## Run

### Examples:
The default prompts are 4, one warmup: "what is OpenVINO?", 3 duplicate 1k prompts for counting the avg. performance.
#### Windows:
`.\build\Release\llm -token .\{YOUR_OWN_RELATIVE_PATH}\openvino_tokenizer.xml) -detoken .\{YOUR_OWN_RELATIVE_PATH}\openvino_detokenizer.xml -m .\{YOUR_OWN_RELATIVE_PATH}\openvino_model.xml --output_fixed_len 256`

## Reduce Logits Optimization
This optimization will modify the graph of OV model IR and largely improve first token latency.
This modified OV IR could also be used with Python native OpenVINO API pipeline.

### Generate modified OV IR:
Adding config `--reduce_logits` will generate a new optimizated LLM model IR `modified_openvino_model.xml` and `modified_openvino_model.bin`.  

`.\build\Release\llm -token .\{YOUR_OWN_RELATIVE_PATH}\openvino_tokenizer.xml) -detoken .\{YOUR_OWN_RELATIVE_PATH}\openvino_detokenizer.xml -m .\{YOUR_OWN_RELATIVE_PATH}\openvino_model.xml --reduce_logits` 

### Run with modified OV IR:
 
`.\build\Release\llm -token .\{YOUR_OWN_RELATIVE_PATH}\openvino_tokenizer.xml) -detoken .\{YOUR_OWN_RELATIVE_PATH}\openvino_detokenizer.xml -m .\{YOUR_OWN_RELATIVE_PATH}\modified_openvino_model.xml --output_fixed_len 256`

## Edit and Debug
The testing prompts are set inside of the llm.cpp. To modify the prompts, open the `.\build\llm.sln` in VS2022, click Solution Explorer(left side) -> llm -> Source Files -> llm.cpp. Edit, save and build in VS2022. Then, run exe again in the Terminal.
Please be careful that `NUM_SENTENCES= 4`should be the same with the real numbers of `std::string sentences`.

## Benchmark Tips
- Clean all other Apps like VS2022 and Browser before benchmarking on the terminal and save the performance output.
- Keep the LLM pipeline running terminal in the front, i.e. don't click other App when benchmarking.
- The pipeline is with model caching by default. Once changing model IR or getting cl kernel error, please delete the `llm-cache` folder before benchmarking. 
- For Python pipeline, reboot PC and benchmark when get cl issue with extra-long prompts
