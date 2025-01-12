import main


def test_end2end(tmp_path):
    main.main(
        [
            "dummy.py",
            f"--save-dir={tmp_path}",
            "--tokenizer-path=openai-community/gpt2",
            "--dataset-type=gsm8k",
        ]
    )
