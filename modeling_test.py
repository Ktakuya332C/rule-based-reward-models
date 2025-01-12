import pytest
import modeling
from transformers import AutoModelForSequenceClassification


def test_save_and_load(tmp_path):
    modeling.RuleBasedRewardConfig.register_for_auto_class()
    modeling.RuleBasedRewardModel.register_for_auto_class("AutoModel")
    modeling.RuleBasedRewardForSequenceClassification.register_for_auto_class(
        "AutoModelForSequenceClassification"
    )
    config = modeling.RuleBasedRewardConfig()
    verifier = modeling.RuleBasedRewardForSequenceClassification(config)
    verifier.save_pretrained(tmp_path)
    _ = AutoModelForSequenceClassification.from_pretrained(
        tmp_path, trust_remote_code=True
    )


@pytest.mark.parametrize(
    argnames=["text", "expected"],
    argvalues=[
        ("This is a pen", None),
        ("There is 3 dogs", "3"),
        ("The answer is -3", "-3"),
        ("Therefore \boxed{-10} is the answer", "-10"),
    ],
)
def test_extract_int(text, expected):
    assert expected == modeling._extract_last_int(text)


def test_end2end_gsm8k():
    config = modeling.RuleBasedRewardConfig(
        tokenizer_path="openai-community/gpt2",
        dataset_type="gsm8k",
    )
    verifier = modeling.RuleBasedRewardForSequenceClassification(config)

    qa1 = (
        "Natalia sold clips to 48 of her friends in April, "
        "and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May? "
        "The answer: 72"
    )
    qa2 = (
        "Weng earns $12 an hour for babysitting. "
        "Yesterday, she just did 50 minutes of babysitting. "
        "How much did she earn? The answer: 11"
    )
    qa3 = (
        "User: Natalia sold clips to 48 of her friends in April, "
        "and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
        "Assistant: The answer: \boxed{72}"
    )
    qa4 = (
        "User: Weng earns $12 an hour for babysitting. "
        "Yesterday, she just did 50 minutes of babysitting. "
        "How much did she earn?"
        "Assistant: The answer: \boxed{11}"
    )
    verifier.model.tokenizer.pad_token = verifier.model.tokenizer.eos_token
    inputs = verifier.model.tokenizer(
        [qa1, qa2, qa3, qa4], return_tensors="pt", padding=True
    )
    output = verifier(**inputs)
    assert output.logits[0][0] == 1.0
    assert output.logits[1][0] == 0.0
    assert output.logits[2][0] == 1.0
    assert output.logits[3][0] == 0.0


@pytest.mark.parametrize(
    argnames=["text", "expected"],
    argvalues=[
        ("This is a \\boxed{3}", "3"),
        ("The answer is -3", None),
        ("There is \\boxed{\\frac{3}{4} + 1}", "\\frac{3}{4}+1"),
    ],
)
def test_extract_boxed(text, expected):
    assert expected == modeling._extract_boxed(text)


def test_end2end_math():
    config = modeling.RuleBasedRewardConfig(
        tokenizer_path="openai-community/gpt2",
        dataset_type="math",
    )
    verifier = modeling.RuleBasedRewardForSequenceClassification(config)

    qa1 = (
        "A plane is expressed parametrically by\n"
        "\\[\\mathbf{v} = \\begin{pmatrix} 1 + s - t \\\\ 2 - s \\\\ 3 - "
        "2s + 2t \\end{pmatrix}.\\]Find the equation of the plane.  Enter "
        "your answer in the form\n"
        "\\[Ax + By + Cz + D = 0,\\]where $A,$ $B,$ $C,$ $D$ are integers "
        "such that $A > 0$ and $\\gcd(|A|,|B|,|C|,|D|) = 1.$"
        "The Answer: \\boxed{2x + z - 5 = 0}"
    )
    qa2 = (
        "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, "
        "\\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]"
        "Find $a+b$ if the piecewise function is continuous "
        "(which means that its graph can be drawn without lifting your pencil from the paper)."
        "The answer: \\boxed{5}"
    )
    qa3 = (
        "User: A plane is expressed parametrically by\n"
        "\\[\\mathbf{v} = \\begin{pmatrix} 1 + s - t \\\\ 2 - s \\\\ 3 - "
        "2s + 2t \\end{pmatrix}.\\]Find the equation of the plane.  Enter "
        "your answer in the form\n"
        "\\[Ax + By + Cz + D = 0,\\]where $A,$ $B,$ $C,$ $D$ are integers "
        "such that $A > 0$ and $\\gcd(|A|,|B|,|C|,|D|) = 1.$"
        "Assistant: The Answer: \\boxed{2x + z - 5 = 0}"
    )
    qa4 = (
        "User: Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, "
        "\\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]"
        "Find $a+b$ if the piecewise function is continuous "
        "(which means that its graph can be drawn without lifting your pencil from the paper)."
        "Assstant: The answer: \\boxed{5}"
    )
    verifier.model.tokenizer.pad_token = verifier.model.tokenizer.eos_token
    inputs = verifier.model.tokenizer(
        [qa1, qa2, qa3, qa4], return_tensors="pt", padding=True
    )
    output = verifier(**inputs)
    assert output.logits[0][0] == 1.0
    assert output.logits[1][0] == 0.0
    assert output.logits[2][0] == 1.0
    assert output.logits[3][0] == 0.0
