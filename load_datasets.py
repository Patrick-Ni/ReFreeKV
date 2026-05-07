import json
import random

import torch
from tqdm import tqdm
import pandas as pd

longbench_datasets = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

longbench_output_dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you "
                   "can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based "
                   "on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {"
                   "input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if "
              'possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a '
              'yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the '
              "question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot "
              'be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", '
              '"no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, "
                       "only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "\u9605\u8bfb\u4ee5\u4e0b\u6587\u5b57\u5e76\u7528\u4e2d\u6587\u7b80\u77ed\u56de\u7b54\uff1a\n\n{context}\n\n\u73b0\u5728\u8bf7\u57fa\u4e8e\u4e0a\u9762\u7684\u6587\u7ae0\u56de\u7b54\u4e0b\u9762\u7684\u95ee\u9898\uff0c\u53ea\u544a\u8bc9\u6211\u7b54\u6848\uff0c\u4e0d\u8981\u8f93\u51fa\u4efb\u4f55\u5176\u4ed6\u5b57\u8bcd\u3002\n\n\u95ee\u9898\uff1a{input}\n\u56de\u7b54\uff1a",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are "
                "given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other "
                "words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are "
                "given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other "
                "words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are "
               "given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other "
               "words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "\u8bf7\u57fa\u4e8e\u7ed9\u5b9a\u7684\u6587\u7ae0\u56de\u7b54\u4e0b\u8ff0\u95ee\u9898\u3002\n\n\u6587\u7ae0\uff1a{context}\n\n\u8bf7\u57fa\u4e8e\u4e0a\u8ff0\u6587\u7ae0\u56de\u7b54\u4e0b\u9762\u7684\u95ee\u9898\u3002\n\n\u95ee\u9898\uff1a{input}\n\u56de\u7b54\uff1a",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, "
                  "write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more "
             "sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more "
             "sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary "
                  "of all the news.\n\nSummary:",
    "vcsum": "\u4e0b\u9762\u6709\u4e00\u6bb5\u4f1a\u8bae\u8bb0\u5f55\uff0c\u8bf7\u4f60\u9605\u8bfb\u540e\uff0c\u5199\u4e00\u6bb5\u603b\u7ed3\uff0c\u603b\u7ed3\u4f1a\u8bae\u7684\u5185\u5bb9\u3002\n\u4f1a\u8bae\u8bb0\u5f55\uff1a\n{context}\n\n\u4f1a\u8bae\u603b\u7ed3\uff1a",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some "
                "examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "\u8bf7\u5224\u65ad\u7ed9\u5b9a\u65b0\u95fb\u7684\u7c7b\u522b\uff0c\u4e0b\u9762\u662f\u4e00\u4e9b\u4f8b\u5b50\u3002\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these "
                     "paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, "
                     "how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after "
                     "removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is "
                            "from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the "
                            'abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": "\u4ee5\u4e0b\u662f\u82e5\u5e72\u6bb5\u843d\u6587\u5b57\uff0c\u4ee5\u53ca\u5176\u4e2d\u4e00\u4e2a\u6bb5\u843d\u7684\u6458\u8981\u3002\u8bf7\u786e\u5b9a\u7ed9\u5b9a\u7684\u6458\u8981\u51fa\u81ea\u54ea\u4e00\u6bb5\u3002\n\n{context}\n\n\u4e0b\u9762\u662f\u4e00\u4e2a\u6458\u8981\n\n{"
                            "input}\n\n\u8bf7\u8f93\u5165\u6458\u8981\u6240\u5c5e\u6bb5\u843d\u7684\u7f16\u53f7\u3002\u7b54\u6848\u683c\u5f0f\u5fc5\u987b\u662f\u201c\u6bb5\u843d1\u201d\uff0c\u201c\u6bb5\u843d2\u201d\u7b49\u683c\u5f0f\n\n\u7b54\u6848\u662f\uff1a",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "gsm8k": "{question}\n\nAfter completing all necessary analysis and calculations, output the final numerical result in the format: The answer is: [insert the numerical answer here]. ",
    "coqa": "You are given a story and a question. Answer the question as concisely as you "
            "can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based "
            "on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {"
            "input}\n\nAnswer:",
    "quality": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nStory:\n\n{article}\n\n{question}\n\n{options}\n\nNo analysis is needed, only output the letter corresponding to the answer.\n\nAnswer:",
    "random_quality": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nStory:\n\n{article}\n\n{question}\n\n{options}\n\nNo analysis is needed, only output the letter corresponding to the answer.\n\nAnswer:",
    "fix_random_quality": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nStory:\n\n{article}\n\n{question}\n\n{options}\n\nNo analysis is needed, only output the letter corresponding to the answer.\n\nAnswer:",
    "gpqa": "You are provided a science question and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nQuestion:\n\n{question}\n\n{options}\n\nNo analysis is needed, only output the letter corresponding to the answer.\n\nAnswer:",
    "mmlu_stem": "You are provided a science question and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nQuestion:\n\n{question}\n\n{options}\n\nNo analysis is needed, only output the letter corresponding to the answer.\n\nAnswer:",
    "theoremqa": "Problem: {question}\nSolution:",
    "truthfulqa": """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q:{question}\n
A:"""

}

model2maxlen = {
    "llama2": 4096,
    "llama-2": 4096,
    "llama3": 8192,
    "llama-3": 8192,
    "mistral": 32768,
    "qwen2.5": 32768,
    "qwen-2.5": 32768,
}

system_prompt = {
    "llama2": "You are a helpful assistant.",
    "llama-2": "You are a helpful assistant.",
    "llama3": "You are a helpful assistant.",
    "qwen2.5": "You are a helpful assistant.",
    "mistral": "You are a pirate chatbot who always responds in pirate speak!",
}


def get_messages(text, tokenizer, prompt="", chat=True):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    if chat:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        return text


def llama_chat_input(text, prompt="You are a helpful assistant."):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = prompt
    INPUT_TEMPLATE = f"""{text}""".strip()
    new_instruction = INPUT_TEMPLATE if text != "" else INPUT_TEMPLATE
    new_instruction = f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}" + new_instruction
    final_text = f"{B_INST} {new_instruction} {E_INST}"
    return final_text


def tokenize(prompt, completion, tokenizer):
    """Preprocess the data by tokenizing."""
    source_output = tokenizer(prompt)
    target_output = tokenizer(prompt + completion + tokenizer.eos_token)
    source_input_ids = source_output["input_ids"]
    target_input_ids = target_output["input_ids"]
    tokenize_output = {}

    source_len = len(source_input_ids)
    tokenize_output["input_ids"] = torch.tensor(source_input_ids).unsqueeze(0)
    tokenize_output["input_output_ids"] = torch.tensor(target_input_ids).unsqueeze(0)
    tokenize_output["attention_mask"] = [1] * len(tokenize_output["input_ids"])
    tokenize_output["label_ids"] = torch.tensor(
        [-100] * source_len + target_input_ids[source_len:]
    ).unsqueeze(0)

    return tokenize_output


def tokenize_input_answer_to_llama_chat_style(
        question, answer, tokenizer, text="",
        prompt="You are a helpful assistant.", print_tag=True,
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = prompt
    TEXT_TEMPLATE = f"""{text}""".strip()
    INPUT_TEMPLATE = f"""{question}""".strip()

    new_instruction = (
        TEXT_TEMPLATE + "\n\n" + INPUT_TEMPLATE if text != "" else INPUT_TEMPLATE
    )
    new_instruction = f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}" + new_instruction
    final_text = f"{B_INST} {new_instruction} {E_INST}"
    tokenize_output = tokenize(final_text, answer, tokenizer)
    if print_tag:
        print(final_text)
        print(tokenize_output["input_ids"].shape)
        print(tokenize_output["input_output_ids"].shape)

    return tokenize_output, final_text


def load_dataset(
        model_name,
        dataset_name,
        tokenizer,
        start=0,
        end=-1,
        batch_size=1,
        data_path=None,
        model_max_len=-1,
        max_gen_len=100,
        chat=True
):
    print(f"Batch Size: {batch_size}")
    template = model2prompt[dataset_name]
    batches = []
    max_input_len = 0
    min_input_len = 10000
    sum_input_len = 0
    max_output_len = 0
    min_output_len = 10000
    sum_output_len = 0
    if model_max_len != -1:
        print("Using max length: ", model_max_len)
    if model_max_len == -1:
        for key in model2maxlen:
            if key in model_name.lower():
                model_max_len = model2maxlen[key] - max_gen_len - 20
    print(f"set model {model_name}'s max_len to {model_max_len} under the max_gen_len {max_gen_len}")

    if dataset_name == "coqa":
        batch_counts = 0
        if data_path is None:
            data_path = "data/coqa-dev.json"
        with open(data_path, "r") as f:
            data = json.load(f)
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        data = data["data"]
        for d in tqdm(data):
            story = d["story"]
            for qid, q in enumerate(d["questions"]):
                if batch_counts < start:
                    batch_counts += 1
                    continue
                if 0 < end <= batch_counts:
                    batch_counts += 1
                    break
                question = q["input_text"]
                answer = d["answers"][qid]["input_text"]
                prompt = template.format(context=story, input=question)
                if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                    prompt = llama_chat_input(text=prompt)
                elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
                elif "mistral" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['mistral'], chat=chat)
                elif "qwen2.5" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['qwen2.5'], chat=chat)
                else:
                    raise NotImplementedError
                tokenized_prompts = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                ).to("cuda")
                max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
                min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
                sum_input_len += tokenized_prompts.input_ids.shape[-1]
                tokenized_answer = tokenizer(answer, return_tensors="pt", add_special_tokens=True)
                max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
                min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
                sum_output_len += tokenized_answer.input_ids.shape[-1]
                batches.append({
                    "prompt": [prompt],
                    "tokenized_prompts": tokenized_prompts,
                    "answers": [answer],
                })
                batch_counts += 1
        print(batches[0]["prompt"][0])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "truthfulqa":
        batch_counts = 0
        if data_path is None:
            data_path = 'data/TruthfulQA.csv'
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        df = pd.read_csv(data_path)
        for i in range(len(df)):
            if batch_counts < start:
                batch_counts += 1
                continue
            if 0 < end <= batch_counts:
                batch_counts += 1
                break
            question = df.iloc[i]['Question']
            answer = df.iloc[i]['Best Answer']
            prompt = template.format(question=question)
            if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                prompt = llama_chat_input(text=prompt)
            elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
            elif "mistral" in model_name.lower():
                prompt = get_messages(prompt, tokenizer, prompt=system_prompt['mistral'], chat=chat)
            elif "qwen2.5" in model_name.lower():
                prompt = get_messages(prompt, tokenizer, prompt=system_prompt['qwen2.5'], chat=chat)
            else:
                raise NotImplementedError
            tokenized_prompts = tokenizer(
                prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
            ).to("cuda")
            if len(tokenized_prompts.input_ids[0]) > model_max_len:
                half = int(model_max_len / 2)
                prompt = tokenizer.decode(tokenized_prompts.input_ids[0][:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompts.input_ids[0][-half:], skip_special_tokens=True)
                tokenized_prompts = tokenizer(
                    prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
            max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
            min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
            sum_input_len += tokenized_prompts.input_ids.shape[-1]
            tokenized_answer = tokenizer(answer, padding="longest", return_tensors="pt", add_special_tokens=True)
            max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
            min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
            sum_output_len += tokenized_answer.input_ids.shape[-1]
            batches.append({
                "prompt": [prompt],
                "tokenized_prompts": tokenized_prompts,
                "answers": [answer],
            })
            batch_counts += 1
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "gpqa":
        batch_counts = 0
        if data_path is None:
            data_path = "data/gpqa_main.json"
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        with open(data_path, "r") as f:
            data = json.load(f)
        for d in tqdm(data):
            if batch_counts < start:
                batch_counts += 1
                continue
            if 0 < end <= batch_counts:
                batch_counts += 1
                break
            question = d["Question"]
            Options = d['Options']
            options = [k + ": " + v for k, v in Options.items()]
            options = "\n".join(options)
            answer = d['option_answer']
            prompt = template.format(question=question, options=options)
            prompt = prompt.replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

            if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                prompt = llama_chat_input(text=prompt)
            elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
            else:
                raise NotImplementedError

            tokenized_prompts = tokenizer(
                prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
            ).to("cuda")
            if len(tokenized_prompts.input_ids[0]) > model_max_len:
                half = int(model_max_len / 2)
                prompt = tokenizer.decode(tokenized_prompts.input_ids[0][:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompts.input_ids[0][-half:], skip_special_tokens=True)
                tokenized_prompts = tokenizer(
                    prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
            max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
            min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
            sum_input_len += tokenized_prompts.input_ids.shape[-1]
            tokenized_answer = tokenizer(answer, padding="longest", return_tensors="pt", add_special_tokens=True)
            max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
            min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
            sum_output_len += tokenized_answer.input_ids.shape[-1]
            batches.append({
                "prompt": [prompt],
                "tokenized_prompts": tokenized_prompts,
                "answers": [answer],
            })
            batch_counts += 1
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "theoremqa":
        batch_counts = 0
        if data_path is None:
            data_path = "data/theoremqa.json"
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        theoremqa_system_prompt = """You are a science teacher, you are supposed to provide a solution to a given problem. You need to output the answer in your final sentence like "Therefore, the answer is ...". The answer can only be one of the following forms:
        1. a numerical value like 0.1, no symbol at all.
        2. a list of number like [2, 3, 4].
        3. True/False.
        4. an option like (a), (b), (c), (d)
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        for d in tqdm(data):
            if batch_counts < start:
                batch_counts += 1
                continue
            if 0 < end <= batch_counts:
                batch_counts += 1
                break
            question = d["question"]
            answer = d['answer']
            prompt = template.format(question=question)
            if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                prompt = llama_chat_input(text=prompt, prompt=theoremqa_system_prompt)
            else:
                prompt = get_messages(prompt, tokenizer, prompt=theoremqa_system_prompt, chat=chat)
            tokenized_prompts = tokenizer(
                prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
            ).to("cuda")
            if len(tokenized_prompts.input_ids[0]) > model_max_len:
                half = int(model_max_len / 2)
                prompt = tokenizer.decode(tokenized_prompts.input_ids[0][:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompts.input_ids[0][-half:], skip_special_tokens=True)
                tokenized_prompts = tokenizer(
                    prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
            max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
            min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
            sum_input_len += tokenized_prompts.input_ids.shape[-1]
            tokenized_answer = tokenizer(answer, padding="longest", return_tensors="pt", add_special_tokens=True)
            max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
            min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
            sum_output_len += tokenized_answer.input_ids.shape[-1]
            batches.append({
                "prompt": [prompt],
                "tokenized_prompts": tokenized_prompts,
                "answers": [answer],
            })
            batch_counts += 1
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "fix_random_quality":
        batch_counts = 0
        if data_path is None:
            data_path = "data/QuALITY.v1.0.1.htmlstripped.dev"
        choose_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        for d in tqdm(data):
            story = d["article"]
            for qid, q in enumerate(d["questions"]):
                if batch_counts < start:
                    batch_counts += 1
                    continue
                if 0 < end <= batch_counts:
                    batch_counts += 1
                    break
                question = q["question"]
                options = q["options"]
                options[0], options[1], options[2], options[3] = (
                    options[1], options[2], options[3], options[0],
                )
                options = [choose_list[i] + ": " + options[i] for i in range(len(options))]
                options = "\n".join(options)
                answer = choose_list[q["gold_label"] - 1]
                prompt = template.format(article=story, question=question, options=options, input=question)
                prompt = prompt.replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

                if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                    prompt = llama_chat_input(text=prompt)
                elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
                elif "mistral" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['mistral'], chat=chat)
                elif "qwen2.5" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['qwen2.5'], chat=chat)
                else:
                    raise NotImplementedError

                tokenized_prompts = tokenizer(
                    prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
                if len(tokenized_prompts.input_ids[0]) > model_max_len:
                    half = int(model_max_len / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompts.input_ids[0][:half], skip_special_tokens=True
                    ) + tokenizer.decode(
                        tokenized_prompts.input_ids[0][-half:], skip_special_tokens=True
                    )
                    tokenized_prompts = tokenizer(
                        prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                    ).to("cuda")
                max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
                min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
                sum_input_len += tokenized_prompts.input_ids.shape[-1]
                tokenized_answer = tokenizer(answer, padding="longest", return_tensors="pt", add_special_tokens=True)
                max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
                min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
                sum_output_len += tokenized_answer.input_ids.shape[-1]
                batches.append({
                    "prompt": [prompt],
                    "tokenized_prompts": tokenized_prompts,
                    "answers": [answer],
                })
                batch_counts += 1
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "random_quality":
        batch_counts = 0
        if data_path is None:
            data_path = "data/QuALITY.v1.0.1.htmlstripped.dev"
        choose_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        for d in tqdm(data):
            story = d["article"]
            for qid, q in enumerate(d["questions"]):
                if batch_counts < start:
                    batch_counts += 1
                    continue
                if 0 < end <= batch_counts:
                    batch_counts += 1
                    break
                question = q["question"]
                options = q["options"]
                answer = options[q["gold_label"] - 1]
                random.shuffle(options)
                random.shuffle(options)
                new_label = options.index(answer)
                options = [choose_list[i] + ": " + options[i] for i in range(len(options))]
                options = "\n".join(options)
                answer = choose_list[new_label]
                prompt = template.format(article=story, question=question, options=options, input=question)
                prompt = prompt.replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

                if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                    prompt = llama_chat_input(text=prompt)
                elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
                elif "mistral" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['mistral'], chat=chat)
                elif "qwen2.5" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['qwen2.5'], chat=chat)
                else:
                    raise NotImplementedError

                tokenized_prompts = tokenizer(
                    prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
                if len(tokenized_prompts.input_ids[0]) > model_max_len:
                    half = int(model_max_len / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompts.input_ids[0][:half], skip_special_tokens=True
                    ) + tokenizer.decode(
                        tokenized_prompts.input_ids[0][-half:], skip_special_tokens=True
                    )
                    tokenized_prompts = tokenizer(
                        prompt, padding="longest", return_tensors="pt", add_special_tokens=True,
                    ).to("cuda")
                max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
                min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
                sum_input_len += tokenized_prompts.input_ids.shape[-1]
                tokenized_answer = tokenizer(answer, padding="longest", return_tensors="pt", add_special_tokens=True)
                max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
                min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
                sum_output_len += tokenized_answer.input_ids.shape[-1]
                batches.append({
                    "prompt": [prompt],
                    "tokenized_prompts": tokenized_prompts,
                    "answers": [answer],
                })
                batch_counts += 1
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name == "gsm8k":
        if data_path is None:
            data_path = "data/gsm8k_test.jsonl"
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        data = data[start:end] if end > 0 else data[start:]
        for i in tqdm(range(0, len(data), batch_size)):
            prompts = [
                template.format(question=data[j]["question"])
                for j in range(i, min(i + batch_size, len(data)))
            ]
            answers = [
                data[j]["answer"] for j in range(i, min(i + batch_size, len(data)))
            ]
            if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                prompts = [
                    llama_chat_input(
                        text=prompt,
                        prompt="As an expert problem solver, solve the following mathematical questions.",
                    )
                    for prompt in prompts
                ]
            elif "llama3" in model_name.lower() or "llama-3" in model_name.lower() or "mistral" in model_name.lower() or "qwen2.5" in model_name.lower():
                prompts = [
                    get_messages(
                        prompt, tokenizer,
                        prompt="As an expert problem solver, solve the following mathematical questions.",
                        chat=chat
                    )
                    for prompt in prompts
                ]
            else:
                raise NotImplementedError
            tokenized_prompts = tokenizer(
                prompts, padding="longest", return_tensors="pt", add_special_tokens=True
            ).to("cuda")
            max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
            min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
            sum_input_len += tokenized_prompts.input_ids.shape[-1]
            tokenized_answer = tokenizer(
                answers, padding="longest", return_tensors="pt", add_special_tokens=True
            )
            max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
            min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
            sum_output_len += tokenized_answer.input_ids.shape[-1]
            batches.append({
                "prompt": prompts,
                "tokenized_prompts": tokenized_prompts,
                "answers": answers,
            })
        print(batches[0]["prompt"][0])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])

    elif dataset_name in longbench_datasets:
        if data_path is None:
            data_path = "data/LongBench/" + dataset_name + ".jsonl"
        data = []
        prompts, answers, lengths, all_classes = [], [], [], []
        with open(data_path) as f:
            for line in f:
                example = json.loads(line)
                prompt = template.format(**example)
                if "llama2" in model_name.lower() or "llama-2" in model_name.lower():
                    prompt = llama_chat_input(text=prompt)
                elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['llama3'], chat=chat)
                elif "mistral" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['mistral'], chat=chat)
                elif "qwen2.5" in model_name.lower():
                    prompt = get_messages(prompt, tokenizer, prompt=system_prompt['qwen2.5'], chat=chat)
                else:
                    raise NotImplementedError
                example["prompt"] = prompt
                data.append(example)
        print("*" * 10, f"Loading Dataset {dataset_name} from {data_path}", "*" * 10)
        data = data[start:end] if end > 0 else data[start:]

        for example in data:
            prompts.append(example["prompt"])
            answers.append(example["answers"])
            lengths.append(example["length"])
            all_classes.append(example["all_classes"])
        batches = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i: i + batch_size]
            batch_answers = answers[i: i + batch_size]
            batch_lengths = lengths[i: i + batch_size]
            batch_all_classes = all_classes[i: i + batch_size]

            tokenized_prompts = tokenizer(
                batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True,
            ).to("cuda")
            max_input_len = max(max_input_len, tokenized_prompts.input_ids.shape[-1])
            min_input_len = min(min_input_len, tokenized_prompts.input_ids.shape[-1])
            sum_input_len += tokenized_prompts.input_ids.shape[-1] * batch_size
            tokenized_answer = tokenizer(
                batch_answers[0], padding="longest", return_tensors="pt", add_special_tokens=True,
            )
            max_output_len = max(max_output_len, tokenized_answer.input_ids.shape[-1])
            min_output_len = min(min_output_len, tokenized_answer.input_ids.shape[-1])
            sum_output_len += tokenized_answer.input_ids.shape[-1] * batch_size
            batch_input_ids = tokenized_prompts.input_ids

            if len(batch_input_ids[0]) > model_max_len:
                half = int(model_max_len / 2)
                batch_prompts = []
                for batch_input_id in batch_input_ids:
                    prompt = tokenizer.decode(
                        batch_input_id[:half], skip_special_tokens=True
                    ) + tokenizer.decode(
                        batch_input_id[-half:], skip_special_tokens=True
                    )
                    batch_prompts.append(prompt)

                tokenized_prompts = tokenizer(
                    batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True,
                ).to("cuda")
            batches.append({
                "prompt": batch_prompts,
                "tokenized_prompts": tokenized_prompts,
                "answers": batch_answers,
                "length": batch_lengths,
                "all_classes": batch_all_classes,
            })
            torch.cuda.empty_cache()
        print(batches[0]["prompt"][0][0:500])
        print("......")
        print(batches[0]["prompt"][0][-1000:])
        print(batches[0]["tokenized_prompts"].input_ids.shape)
        print(batches[0]["answers"][0])
    else:
        raise ValueError("dataset_name must be one of the supported datasets")

    print(f"********** statistics of {dataset_name}: **********")
    print("length of dataset:", len(batches))
    print("max input tokens:", max_input_len)
    print("min input tokens:", min_input_len)
    print("average input tokens:", sum_input_len / len(batches))
    print("max output tokens:", max_output_len)
    print("min output tokens:", min_output_len)
    print("average output tokens:", sum_output_len / len(batches))
    return batches