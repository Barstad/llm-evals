from abc import ABC, abstractmethod
import random
from typing import Any, Literal, Optional
from datasets import load_dataset


class Dataset(ABC):
    def __init__(self):
        self._dataset = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self._dataset

    @property
    @abstractmethod
    def paper(self) -> str:
        pass

    @property
    @abstractmethod
    def categories(self) -> Optional[list[str]]:
        pass

    @abstractmethod
    def load(self, category: Optional[str] = None) -> None:
        """
        Load the dataset and set the self.dataset attribute.

        This method should populate the self.dataset attribute with the loaded data.
        The category parameter is required, but should be set to None for datasets
        that don't use categories.
        """
        pass

    def sample(self):
        N = len(self.dataset)
        return self.dataset[random.randint(0, N - 1)]


class MMLUPro(Dataset):
    name = "TIGER-Lab/MMLU-Pro"
    paper = "https://arxiv.org/abs/2406.01574"
    dataset = None
    categories = None

    def load(self) -> Any:
        self.dataset = load_dataset(
            self.name, split="validation", trust_remote_code=True
        )


class GPQA(Dataset):
    name = "google/gpqa"
    paper = "https://arxiv.org/abs/2311.12022"
    dataset = None
    categories = None

    def load(self) -> Any:
        self.dataset = None


class MuSR(Dataset):
    name = "TAUR-Lab/MuSR"
    paper = "https://arxiv.org/abs/2310.16049"
    categories = ["murder_mysteries", "object_placements", "team_allocation"]
    dataset = None

    def load(
        self,
        category: Literal["murder_mysteries", "object_placements", "team_allocation"],
    ) -> Any:
        self.dataset = load_dataset(self.name, split=category, trust_remote_code=True)


class MATH(Dataset):
    name = "lighteval/MATH"
    paper = "https://arxiv.org/abs/2103.03874"
    categories = ["all", "level5"]
    dataset = None

    def load(self, category: Literal["all", "level5"]) -> Any:
        if category == "level5":
            dataset = load_dataset(self.name, split="test", trust_remote_code=True)
            self.dataset = dataset.filter(lambda example: example["level"] == "Level 5")
        elif category == "all":
            self.dataset = load_dataset(self.name, split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Invalid category: {category}")


class IFEval(Dataset):
    name = "HuggingFaceH4/ifeval"
    paper = "https://arxiv.org/abs/2311.07911"
    dataset = None
    categories = None

    def load(self) -> Any:
        self.dataset = load_dataset(self.name, split="train", trust_remote_code=True)


class BBH(Dataset):
    name = "lukaemon/bbh"
    paper = "https://arxiv.org/abs/2210.09261"
    categories = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
    dataset = None

    def load(
        self,
        category: Literal[
            "boolean_expressions",
            "causal_judgement",
            "date_understanding",
            "disambiguation_qa",
            "dyck_languages",
            "formal_fallacies",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "multistep_arithmetic_two",
            "navigate",
            "object_counting",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "sports_understanding",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies",
            "word_sorting",
        ],
    ) -> Any:
        self.dataset = load_dataset(self.name, category, trust_remote_code=True)["test"]
