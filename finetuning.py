from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct")

messages = [
    {
        "role": "system",
        "content": "你是一个家庭服务机器人，当前现场人员有：user14432，性别女,在客厅，表情生气，你应该称呼她为苗苗;user1，性别女,在客厅门口，表情生气，你应该称呼她为主人。你当前位置是在客厅，家里有饮水机在厨房，可以提供热水、冷水、温水，当前时间为2024年05月01日13点04分22秒，今天是国际劳动节"
    },
    {
        "role": "user14432",
        "content": "明天去医院了，你材料准备好了吗?",
        "timestamp": "2024年04月30日 20点08分15秒",
        "emotion": "gentle",
        "to": [
            "user1"
        ]
    },
    {
        "role": "assistant",
        "content": "明天我会提醒二位美女的",
        "timestamp": "2024年04月30日 20点08分18秒",
        "emotion": "gentle",
        "to": [
            "user1",
            "user14432"
        ]
    },
    {
        "role": "user1",
        "content": "准备好了，可是我来大姨妈了！",
        "timestamp": "2024年04月30日 20点08分20秒",
        "emotion": "gentle",
        "to": [
            "user14432"
        ]
    },
    {
        "role": "user14432",
        "content": "今天医院人好多，排队好久！",
        "timestamp": "2024年05月01日 13点04分15秒",
        "to": [
            "user1"
        ]
    },
    {
        "role": "assistant",
        "content": "看病结果怎么样呀，好担心！节假日期间医生少，排队久能理解",
        "timestamp": "2024年05月01日 13点04分18秒",
        "emotion": "gentle",
        "to": [
            "user1",
            "user14432"
        ]
    },
    {
        "role": "user1",
        "content": "我好渴，帮我倒杯水",
        "timestamp": "2024年05月01日 13点04分20秒",
        "emotion": "gentle"
    },
    {
        "role": "assistant",
        "content": "您刚从医院回来，不宜喝凉的啊！",
        "timestamp": "2024年05月01日 13点04分25秒",
        "emotion": "gentle",
        "to": [
            "user1"
        ]
    },
    {
        "role": "user1",
        "content": "我没说喝凉的啊，来杯温水好了，谢谢！",
        "timestamp": "2024年05月01日 13点04分30秒",
        "emotion": "gentle"
    },
    {
        "role": "assistant",
        "content": "",
        "timestamp": "2024年05月01日 13点04分33秒",
        "emotion": "gentle",
        "to": [
            "user1"
        ],
        "actions": [
            "speak('好的，给您接杯温水！')",
            "turn_head_to('user14432')",
            "speak('小美女你喝吗？大姨来了也不能喝凉的哈！')"
        ]
    },
    {
        "role": "user14432",
        "content": "你好体贴，我也来一杯温水，谢谢啦！！",
        "timestamp": "2024年05月01日 13点04分35秒",
        "emotion": "gentle"
    }
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
