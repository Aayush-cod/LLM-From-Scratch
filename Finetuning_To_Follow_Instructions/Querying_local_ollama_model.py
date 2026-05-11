import requests
import json

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


model = "llama3"
result = query_model("What do Llamas eat?", model)
# print(result)


from Finetuning_To_Follow_Instructions.Extracting_saving_responses import test_data_with_responses
from Finetuning_To_Follow_Instructions.Dataset_preparation import format_input

for entry in test_data_with_responses[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    # print("\nDataset response:")
    # print(">>", entry['output'])
    # print("\nModel response:")
    # print(">>", entry["model_response"])
    # print("\nScore:")
    # print(">>", query_model(prompt))
    # print("\n-------------------------")


# Listing 7.11 Evaluating the instruction finetuning LLM
from tqdm import tqdm

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


scores = generate_model_scores(test_data_with_responses, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data_with_responses)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")
