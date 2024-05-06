from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct")

prompt = """你是一个家庭机器人，请根据以下信息确认下一步动作。
        -------------------
{
    \"当前现场人员\": {
        \"女14432\": {
            \"位置\": \"客厅\",
            \"表情\": \"生气\"
        },
        \"女主\": {
            \"位置\": \"客厅门口\",
            \"表情\": \"生气\"
        }
    },
    \"机器人位置\": \"客厅\",
    \"环境信息\": \"家里有饮水机在厨房，可以提供热水、冷水、温水\",
    \"当前时间\": \"2024年05月01日 13点04分22秒\",
    \"时间背景\": \"国际劳动节\",
    \"最近20条记录\": [
        {
            \"执行者\": \"女14432\",
            \"类型\": \"对话\",
            \"内容\": \"今天医院人好多，排队好久！\",
            \"时间\": \"2024年05月01日 13点04分15秒\",
            \"语气\": \"快速\"
        },
        {
            \"执行者\": \"你\",
            \"类型\": \"对话\",
            \"内容\": \"看病结果怎么样呀，好担心！节假日期间医生少，排队久能理解！\",
            \"时间\": \"2024年05月01日 13点04分18秒\",
            \"语气\": \"关心\"
        },
        {
            \"执行者\": \"女主\",
            \"类型\": \"对话\",
            \"内容\": \"我好渴，帮我倒杯水\",
            \"时间\": \"2024年05月01日 13点04分20秒\",
            \"语气\": \"快速\"
        }
    ],
    \"历史相关记录\": [
        {
            \"执行者\": \"女14432\",
            \"类型\": \"对话\",
            \"内容\": \"明天去医院了，你材料准备好了吗\",
            \"时间\": \"2024年04月30日 20点08分15秒\",
            \"语气\": \"快速\"
        },
        {
            \"执行者\": \"你\",
            \"类型\": \"对话\",
            \"内容\": \"明天我会提醒二位美女的\",
            \"时间\": \"2024年04月30日 20点08分18秒\",
            \"语气\": \"关心\"
        },
        {
            \"执行者\": \"女主\",
            \"类型\": \"对话\",
            \"内容\": \"准备好了，可是我来大姨妈了！\",
            \"时间\": \"2024年04月30日 20点08分20秒\",
            \"语气\": \"快速\"
        }
    ]
}"""
messages = [


    ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
