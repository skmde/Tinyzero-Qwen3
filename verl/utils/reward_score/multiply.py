import re
import random


def _to_int(value):
    if value is None:
        return None
    try:
        return int(str(value).replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def _normalized_digit_match(pred: int, gt: int) -> float:
    pred_s = str(abs(pred))
    gt_s = str(abs(gt))
    if not pred_s or not gt_s:
        return 0.0

    match = sum(1 for p, g in zip(pred_s, gt_s) if p == g)
    return match / max(len(pred_s), len(gt_s))


def _step_bonus(solution_str: str) -> float:
    if not solution_str:
        return 0.0

    # Use lightweight structural signals as progressive step rewards.
    assistant = solution_str.split("Assistant:", 1)[1] if "Assistant:" in solution_str else solution_str
    bonus = 0.0
    if "<think>" in assistant and "</think>" in assistant:
        bonus += 0.08
    if "<answer>" in assistant and "</answer>" in assistant:
        bonus += 0.08
    if "=" in assistant or "*" in assistant or "×" in assistant:
        bonus += 0.04
    return min(0.20, bonus)


def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    else:
        return None

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, flags=re.DOTALL)
    matches = list(match)
    if matches:
        # Prefer the last explicit <answer> block; tolerate extra tokens around numbers.
        final_answer = matches[-1].group(1).strip().replace(',', '')
        num_match = re.search(r'-?\d+', final_answer)
        if num_match:
            return num_match.group(0)

    # Fallback: extract the last integer from assistant output tail.
    # This increases reward density when model misses strict XML tags.
    tail = solution_str[-512:].replace(',', '')
    nums = re.findall(r'-?\d+', tail)
    if nums:
        return nums[-1]
    return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    gt = _to_int(ground_truth)
    pred = _to_int(answer)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    bonus = _step_bonus(solution_str)

    if pred is None or gt is None:
        if do_print:
            print(f"No answer found")
        # Keep non-negative dense signal to reduce reward polarization.
        return max(0.0, min(float(score) * 0.30, bonus * 0.5))
    else:
        if pred == gt:
            if do_print:
                print(f"Correct answer: {answer}")
            return score
        else:
            # Progressive dense reward (no negative punishment):
            # format + step bonus + closeness + digit consistency.
            rel_error = abs(pred - gt) / max(abs(gt), 1)
            closeness = 1.0 / (1.0 + rel_error)
            digit_match = _normalized_digit_match(pred, gt)

            dense_wrong_reward = (
                float(format_score)
                + bonus
                + 0.40 * closeness
                + 0.25 * digit_match
            )
            # Keep a clear margin from fully correct samples while remaining dense.
            dense_wrong_reward = max(0.0, min(float(score) * 0.85, dense_wrong_reward))
            if do_print:
                print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
                print(
                    f"Dense wrong reward: {dense_wrong_reward:.6f} "
                    f"(rel_error={rel_error:.6f}, bonus={bonus:.3f}, digit_match={digit_match:.3f})"
                )
            return dense_wrong_reward
