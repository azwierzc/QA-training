import logging
from typing import Dict
import json
import torch
from tqdm.auto import tqdm
import logging
import os
import sys
from dataclasses import dataclass
from typing import List
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from QA_training.data.utils_data import load_dataset, select_samples
from evaluation import evaluate
logger = logging.getLogger(__name__)

with open("args.json", "r", encoding="utf-8") as j:
    args = json.load(j)

question_answering_column_name_mapping = {
    "squad_v2": ("question", "context", "answer"),
}


def main():
    set_seed(args["seed"])
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    training_args = parser.parse_dict(args)[0]

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir)
            and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # ============================================= SETTING MODEL AND DATASET =============================================
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    config = AutoConfig.from_pretrained(args["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model_name"], from_tf=bool(".ckpt" in args["model_name"]), config=config)
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    dataset = load_dataset("./../data/" + args["train_file"], seed=args["seed"])

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    # ============================================= PREPARING FEATURES =============================================
    max_input_length = min(args["max_input_length"], tokenizer.model_max_length)
    max_target_length = min(args["max_target_length"], tokenizer.model_max_length)
    # process the examples in input and target text format and the eos token at the end
    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
        example['target_text'] = '%s </s>' % example['generative_answer']
        return example

    # tokenize the examples
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=max_input_length)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=max_target_length)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }
        return encodings


    train_dataset = select_samples(dataset["train"], args["max_train_samples"])
    train_dataset = train_dataset.map(add_eos_to_examples)
    train_dataset = train_dataset.map(convert_to_features, batched=True, remove_columns=train_dataset.column_names)

    val_examples = select_samples(dataset["val"], args["max_val_samples"])
    val_dataset = val_examples.map(add_eos_to_examples)
    val_dataset = val_dataset.map(convert_to_features, batched=True, remove_columns=val_dataset.column_names)

    test_examples = select_samples(dataset["test"], args["max_predict_samples"])
    test_dataset = test_examples.map(add_eos_to_examples)
    test_dataset = test_dataset.map(convert_to_features, batched=True, remove_columns=test_dataset.column_names)

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)


    # ============================================= SETTING TRAINER =============================================
    # prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
    # this is necessacry because the trainer directly passes this dict as arguments to the model
    # so make sure the keys match the parameter names of the forward method
    @dataclass
    class T2TDataCollator():
        def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
            """
            Take a list of samples from a Dataset and collate them into a batch.
            Returns:
                A dictionary of tensors
            """
            input_ids = torch.stack([example['input_ids'] for example in batch])
            lm_labels = torch.stack([example['target_ids'] for example in batch])
            lm_labels[lm_labels[:, :] == 0] = -100
            attention_mask = torch.stack([example['attention_mask'] for example in batch])
            decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])


            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': lm_labels,
                'decoder_attention_mask': decoder_attention_mask
            }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=T2TDataCollator()
    )


    # ============================================= TRAINING =============================================
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            args["max_train_samples"] if args["max_train_samples"] is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # ============================================= EVALUATION =============================================
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = args["max_val_samples"] if args["max_val_samples"] is not None else len(val_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # ============================================= PREDICTION =============================================
    if training_args.do_predict:
        test_examples = select_samples(dataset["test"], args["max_val_samples"])
        test_dataset = test_examples.map(add_eos_to_examples)
        test_dataset = test_dataset.map(convert_to_features, batched=True)
        test_dataset.set_format(type='torch', columns=columns)

        logger.info("*** Predict ***")

        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args["per_device_eval_batch_size"])
        answers = []
        for batch in tqdm(dataloader):
            outs = model.generate(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  max_length=max_target_length,
                                  early_stopping=True)
            outs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            answers.extend(outs)

        predictions = []
        references = []
        for ref, pred in zip(test_examples, answers):
            predictions.append(pred)
            if ref["is_impossible"]:
                references.append("")
            else:
                references.append(ref['generative_answer'])

        metrics = evaluate(references, predictions)
        print(metrics)


if __name__ == "__main__":
    main()
