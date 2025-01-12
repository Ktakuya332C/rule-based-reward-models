import sys
import modeling
import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--dataset-type", type=str)
    return parser.parse_args(argv[1:])


def main(argv):
    args = get_args(argv)

    modeling.RuleBasedRewardConfig.register_for_auto_class()
    modeling.RuleBasedRewardModel.register_for_auto_class("AutoModel")
    modeling.RuleBasedRewardForSequenceClassification.register_for_auto_class(
        "AutoModelForSequenceClassification"
    )
    config = modeling.RuleBasedRewardConfig(
        tokenizer_path=args.tokenizer_path,
        dataset_type=args.dataset_type,
    )
    verifier = modeling.RuleBasedRewardForSequenceClassification(config)
    verifier.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main(sys.argv)
