import os
import sys
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizerFast


DEFAULT_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]


class BertBiLSTMForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)
        return {"logits": logits}


def fix_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_label_mapping(model) -> Dict[int, str]:
    config = model.config
    raw_id2label = getattr(config, "id2label", {}) or {}
    mapped = {}

    for key, value in raw_id2label.items():
        try:
            mapped[int(key)] = value
        except (TypeError, ValueError):
            continue

    labels_look_placeholder = (
        len(mapped) == config.num_labels
        and all(isinstance(v, str) and v.startswith("LABEL_") for v in mapped.values())
    )

    if labels_look_placeholder and config.num_labels == len(DEFAULT_LABELS):
        mapped = {idx: label for idx, label in enumerate(DEFAULT_LABELS)}
        config.id2label = mapped
        config.label2id = {label: idx for idx, label in mapped.items()}

    return mapped


def extract_entities(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = 128,
) -> List[Dict[str, object]]:
    model.eval()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded)["logits"]
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    entities: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    for pred_id, (start, end) in zip(pred_ids, offsets):
        if start == end:
            continue

        label = id2label.get(int(pred_id), "O")
        if label == "O" or "-" not in label:
            if current:
                current["word"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if current:
                current["word"] = text[current["start"]:current["end"]]
                entities.append(current)
            current = {"start": start, "end": end, "type": entity_type}
            continue

        if prefix == "I":
            if current and current["type"] == entity_type and start <= current["end"]:
                current["end"] = end
            elif current and current["type"] == entity_type and text[current["end"]:start].strip() == "":
                current["end"] = end
            else:
                if current:
                    current["word"] = text[current["start"]:current["end"]]
                    entities.append(current)
                current = {"start": start, "end": end, "type": entity_type}

    if current:
        current["word"] = text[current["start"]:current["end"]]
        entities.append(current)

    return entities


def pretty_print_entities(text: str, entities: List[Dict[str, object]]) -> None:
    print(f"Input: {text}")
    if not entities:
        print("No entities found.")
        return

    print("Entities:")
    for entity in entities:
        print(
            f"- {entity['type']}: {entity['word']} "
            f"[start={entity['start']}, end={entity['end']}]"
        )


def main() -> None:
    fix_console_encoding()

    model_path = "./final_model"
    if not os.path.isdir(model_path):
        print(f"Model directory not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertBiLSTMForNER.from_pretrained(model_path).to(device)
    id2label = build_label_mapping(model)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
    else:
        text = input("Please enter a sentence: ").strip()

    if not text:
        print("Empty input, nothing to predict.")
        return

    entities = extract_entities(text, model, tokenizer, id2label, device)
    pretty_print_entities(text, entities)


if __name__ == "__main__":
    main()
