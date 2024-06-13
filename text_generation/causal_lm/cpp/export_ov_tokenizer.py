import os
import openvino as ov
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import argparse
import warnings
from transformers import AutoTokenizer
from openvino import save_model
from openvino_tokenizers import convert_tokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        default='meta-llama/Llama-2-7b-hf',
                        required=False,
                        type=str,
                        help='orignal model path')
    parser.add_argument('-o',
                        '--output',
                        default='./fp16_model',
                        required=False,
                        type=str,
                        help='Required. path to save the ir model')
    args = parser.parse_args()

    ir_model_path = Path(args.output)
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)
    ir_model_file = ir_model_path / "openvino_model.xml"



    print(" --- exporting tokenizer --- ")
    # https://github.com/openvinotoolkit/openvino_tokenizers
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    hf_tokenizer.save_pretrained(ir_model_path)
    # ov_tokenizer = convert_tokenizer(hf_tokenizer)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    save_model(ov_tokenizer, ir_model_path / "openvino_tokenizer.xml")
    save_model(ov_detokenizer, ir_model_path / "openvino_detokenizer.xml")
    # copy 2 IR into the folder where stores the LLM IR(.xml)

