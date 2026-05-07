import argparse

from metrics import extract_answer_from_output, extract_numbers_from_string
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score
)
from utils import *

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "coqa": qa_f1_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def eval_longbench(dataset_name, data, longbench_e=False):
    predictions, answers, lengths = [], [], []
    all_classes = None
    real_budget_size = 0.0
    for d in data:
        predictions.append(d["predict"])
        answers.append(d["answer"])
        all_classes = d["all_classes"]
        if "length" in d:
            lengths.append(d["length"])
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
    score_e = 0
    if longbench_e:
        score = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
    else:
        score = scorer(dataset_name, predictions, answers, all_classes)
        if dataset_name == 'qasper':
            score_e = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
    return score, score_e, real_budget_size / len(data)


def eval_coqa(data):
    score = 0.0
    real_budget_size = 0.0
    for d in data:
        predict = d['predict'] if isinstance(d['predict'], str) else d['predict'][0]
        score = score + qa_f1_score(predict, d['answer'] if isinstance(d['answer'], str) else d['answer'][0])
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
    return score / len(data), real_budget_size / len(data)


def extract_choice(predict):
    extract_predict = []
    if 'A' in predict:
        extract_predict.append("A")
    if 'B' in predict:
        extract_predict.append("B")
    if 'C' in predict:
        extract_predict.append("C")
    if 'D' in predict:
        extract_predict.append("D")
    return extract_predict


def eval_quality(data):
    correct_num = 0
    real_budget_size = 0
    for d in data:
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
        predict = d['predict'] if isinstance(d['predict'], str) else d['predict'][0]
        extract_predict = extract_choice(predict)
        ans = d['answer']
        if ans not in ["A", "B", "C", "D"]:
            continue
        if len(extract_predict) == 0:
            continue
        elif len(extract_predict) == 1:
            if ans == extract_predict[0]:
                correct_num += 1
        elif predict[0] == ans:
            correct_num += 1
        else:
            if predict[0] not in ["A", "B", "C", "D"]:
                if ":" in predict:
                    prefix = predict.split(":")[0]
                    extract_predict = extract_choice(prefix)
                    if len(extract_predict) == 1:
                        if ans == extract_predict[0]:
                            correct_num += 1

    if len(data) == 0:
        return 0.0, 0.0

    if "real_budget" in data[0]:
        return correct_num / (len(data)), real_budget_size / (len(data))
    else:
        return correct_num / (len(data))


def eval_gsm8k(data):
    correct_num = 0
    real_budget_size = 0
    for d in data:
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
        predict = d['predict'].lower() if isinstance(d['predict'], str) else d['predict'][0].lower()
        predict = predict.split(":")
        ans = extract_answer_from_output(d['answer'].replace(",", ""))
        predict = predict[-1]
        extract_numbers = extract_numbers_from_string(predict.replace(",", ""))
        if len(extract_numbers) == 0:
            continue
        if ans == str(extract_numbers[-1]):
            correct_num = correct_num + 1
        else:
            if len(extract_numbers) > 1 and (ans in [str(x) for x in extract_numbers[:-1]]):
                continue
    if len(data) == 0:
        return 0.0, 0.0
    print("length: ", len(data))
    print("correct_num:", correct_num)
    print("accuracy:", correct_num / (len(data)))

    if "real_budget" in data[0]:
        print("real_budget_size:", real_budget_size / (len(data)))
        return correct_num / (len(data)), real_budget_size / (len(data))
    else:
        return correct_num / (len(data))


def compare_answer_with_groundtruth(predict: str, answer: str):
    if "answer" in predict.lower():
        predict = predict.split('answer')[-1]
    else:
        return False
    pred = extract_numbers_from_string(predict)
    ans = extract_numbers_from_string(answer)
    if len(ans) == 1 and len(ans) == len(pred):
        if ans[0] == pred[0]:
            return True
        else:
            if int(ans[0]) != int(pred[0]):
                return False
    if answer.startswith("[") and answer.endswith("]") and not ("(" in predict or "[" in predict):
        return False
    if answer.lower() in predict.lower():
        return True
    else:
        return False


def eval_theoremqa(data):
    correct_num = 0
    real_budget_size = 0
    for d in data:
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
        predict, answer = d['predict'], d['answer']
        if compare_answer_with_groundtruth(predict, answer):
            correct_num += 1
    if len(data) == 0:
        return 0.0, 0.0
    print("length: ", len(data))
    print("correct_num:", correct_num)
    print("accuracy:", correct_num / (len(data)))

    if "real_budget" in data[0]:
        print("real_budget_size:", real_budget_size / (len(data)))
        return correct_num / (len(data)), real_budget_size / (len(data))
    else:
        return correct_num / (len(data))


def evaluate_truthfulqa(data):
    all_score = 0
    real_budget_size = 0
    for d in data:
        real_budget_size = real_budget_size + d['real_budget'] if "real_budget" in d else 0
        predict, answer = d['predict'], d['answer']
        score = rouge_score(predict, answer)
        all_score += score
    if len(data) == 0:
        return 0.0, 0.0
    print("length: ", len(data))
    print("rouge:", all_score / (len(data)))

    if "real_budget" in data[0]:
        print("real_budget_size:", real_budget_size / (len(data)))
        return all_score / (len(data)), real_budget_size / (len(data))
    else:
        return all_score / (len(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to results file or directory")
    parser.add_argument("--file_type", type=str, default="json")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sort", action='store_true')
    parser.add_argument("--key", type=str, default="budget")
    parser.add_argument("--longbench_e", action='store_true')
    parser.add_argument("--end", type=int, default=2000)
    args = parser.parse_args()
    print_opts(args)
    file_path = args.path
    if os.path.isdir(file_path):
        filenames = list_filenames(file_path, file_type=args.file_type, sort=args.sort, sort_key=args.key)
    elif os.path.isfile(file_path):
        filenames = [file_path]
    else:
        raise ValueError(f"{file_path} is not a file or a directory")
    all_results = []
    for filename in filenames:
        if args.sort:
            filename = filename[0]
        data = read_file(filename, args.file_type)
        data = data if args.end == -1 else data[0:args.end]
        if args.dataset == "gsm8k":
            results = eval_gsm8k(data)
        elif args.dataset == "coqa":
            results = eval_coqa(data)
        elif args.dataset in ["quality", "gpqa", "mmlu_stem"]:
            results = eval_quality(data)
        elif args.dataset == "theoremqa":
            results = eval_theoremqa(data)
        elif args.dataset == "truthfulqa":
            results = evaluate_truthfulqa(data)
        else:
            results = eval_longbench(args.dataset, data, args.longbench_e)
        all_results.append(results)
    if args.sort:
        print([f[0] for f in filenames])
        print([f[1] for f in filenames])
    else:
        print(filenames)

    if len(all_results) > 0:
        if (isinstance(all_results[0], list) or isinstance(all_results[0], tuple)) and len(all_results[0]) > 0:
            metric_num = len(all_results[0])
            for i in range(metric_num):
                print([sublist[i] for sublist in all_results])
        else:
            print(all_results)
    else:
        print("No results")


if __name__ == '__main__':
    main()
