import re
import torch
import itertools
from datasets import load_dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer


class RuleBasedRewardConfig(PretrainedConfig):

    def __init__(
        self,
        tokenizer_path="openai-community/gpt2",
        dataset_type="gsm8k",
        **kwargs,
    ):
        self.tokenizer_path = tokenizer_path
        self.dataset_type = dataset_type
        super().__init__(**kwargs)


class RuleBasedRewardModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.idxs = _get_idxs(self.config.dataset_type)

    def forward(self, input_ids, *args, **kwargs):
        batch_size, sequence_length = input_ids.shape
        query_responses = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        rewards = _judge(self.config.dataset_type, self.idxs, query_responses)
        assert len(rewards.shape) == 1 and rewards.shape[0] == batch_size
        hidden_state = (
            rewards.unsqueeze(-1).expand(batch_size, sequence_length).unsqueeze(-1)
        )
        hidden_state = hidden_state.to(input_ids.device)
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=(hidden_state,),
        )


class RuleBasedRewardForSequenceClassification(PreTrainedModel):
    config_class = RuleBasedRewardConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.model = RuleBasedRewardModel(config)
        self.score = torch.nn.Identity()

    def forward(self, input_ids, *args, **kwargs):
        outputs = self.model(input_ids)
        logits = self.score(outputs.last_hidden_state)
        pooled_logits = logits[:, -1, :]
        return SequenceClassifierOutput(logits=pooled_logits)


def _get_idxs(dataset_type):
    if dataset_type == "gsm8k":
        return _get_idxs_gsm8k()
    if dataset_type == "math":
        return _get_idxs_math()
    msg = f"dataset_type {dataset_type} not found"
    raise ValueError(msg)


def _judge(dataset_type, idxs, query_responses):
    if dataset_type == "gsm8k":
        return _judge_gsm8k(idxs, query_responses)
    if dataset_type == "math":
        return _judge_math(idxs, query_responses)
    msg = f"dataset_type {dataset_type} not found"
    raise ValueError(msg)


def _extract_last_int(text):
    m = re.findall(r"-?\d+", text)
    return m[-1] if m else None


def _get_idxs_gsm8k():
    dataset = load_dataset(path="openai/gsm8k", name="main")
    dataset = dataset.map(lambda inst: {"answer": _extract_last_int(inst["answer"])})
    return dataset


def _judge_gsm8k(idxs, query_responses):
    train_dataset, test_dataset = idxs["train"], idxs["test"]
    rewards = torch.zeros(len(query_responses))
    for idx, query_response in enumerate(query_responses):
        for data in itertools.chain(train_dataset, test_dataset):
            if data["question"] in query_response:
                candidate = _extract_last_int(query_response)
                if (
                    candidate is not None
                    and data["answer"] is not None
                    and candidate == data["answer"]
                ):
                    rewards[idx] = 1.0
                break
    return rewards


def _extract_boxed(text):
    m = re.findall(r"boxed\{(.+)\}", text)
    return m[-1].replace(" ", "") if m else None


def _get_idxs_math():
    dataset = load_dataset(path="hendrycks/competition_math")
    dataset = dataset.map(
        lambda inst: {"answer": _extract_boxed(inst["solution"])},
        remove_columns=[
            "level",
            "type",
            "solution",
        ],
    )
    return dataset


def _judge_math(idxs, query_responses):
    train_dataset, test_dataset = idxs["train"], idxs["test"]
    rewards = torch.zeros(len(query_responses))
    for idx, query_response in enumerate(query_responses):
        for data in itertools.chain(train_dataset, test_dataset):
            if data["problem"] in query_response:
                candidate = _extract_boxed(query_response)
                if (
                    candidate is not None
                    and data["answer"] is not None
                    and candidate == data["answer"]
                ):
                    rewards[idx] = 1.0
                break
    return rewards
