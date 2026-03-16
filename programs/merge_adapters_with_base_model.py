#skript for merging lora adapters with the base model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapter(adapter_path, base_model_id, output_dir):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    model = PeftModel.from_pretrained(model, adapter_path)

    model = model.merge_and_unload()

    output_dir = "./merged-model"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

#example:
#merge_adapter("data\models\BitAgent_Bounty_8B_adapter", "BitAgent/BitAgent_Bounty_8B", "data\models\BitAgent_Bounty_8B_merged")