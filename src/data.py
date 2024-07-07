from abc import ABC, abstractmethod
import random
from typing import Any, Literal, Optional
from datasets import load_dataset
import pandas as pd

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
            dataset = load_dataset(self.name, split="train", trust_remote_code=True)
            self.dataset = dataset.filter(lambda example: example["level"] == "Level 5")
        elif category == "all":
            self.dataset = load_dataset(self.name, split="train", trust_remote_code=True)
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


class MMLU(Dataset):
    name = "cais/mmlu"
    paper = "https://arxiv.org/abs/2009.03300"
    dataset = None
    categories = [
         'all', 'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    def load(self, category:Literal['all', 'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'] = 'all') -> Any:
        self.dataset = load_dataset(self.name, category, trust_remote_code=True)["test"]


class AI12ARC(Dataset):
    name = "allenai/ai2_arc"
    paper = "https://arxiv.org/abs/1803.05457"
    dataset = None
    categories = ['ARC-Challenge', 'ARC-Easy']
    def load(self, category:Literal['ARC-Challenge', 'ARC-Easy']) -> Any:
        self.dataset = load_dataset(self.name, category, trust_remote_code=True)["test"]

class GSM8K(Dataset):
    name = "openai/gsm8k"
    paper = "https://arxiv.org/abs/2110.14168"
    dataset = None
    categories = ['main', 'socratic']
    def load(self, category:Literal['main', 'socratic']) -> Any:
        self.dataset = load_dataset(self.name, category, trust_remote_code=True)["test"]


class AGIEval(Dataset):
    name = None
    paper ="https://arxiv.org/abs/2304.06364"
    dataset = None
    categories = [
        "aqua-rat",
        "gaokao-biology",
        "gaokao-chemistry",
        "gaokao-chinese",
        "gaokao-english",
        "gaokao-geography",
        "gaokao-history",
        "gaokao-mathcloze",
        "gaokao-mathqa",
        "gaokao-physics",
        "jec-qa-ca",
        "jec-qa-kd",
        "logiqa-en",
        "logiqa-zh",
        "lsat-ar",
        "lsat-lr",
        "lsat-rc",
        "math",
        "sat-en-without-passage",
        "sat-en",
        "sat-math"
        ]
    
    def load(
            self,
            category:Literal[
                "aqua-rat",
                "gaokao-biology",
                "gaokao-chemistry",
                "gaokao-chinese",
                "gaokao-english",
                "gaokao-geography",
                "gaokao-history",
                "gaokao-mathcloze",
                "gaokao-mathqa",
                "gaokao-physics",
                "jec-qa-ca",
                "jec-qa-kd",
                "logiqa-en",
                "logiqa-zh",
                "lsat-ar",
                "lsat-lr",
                "lsat-rc",
                "math",
                "sat-en-without-passage",
                "sat-en",
                "sat-math"
                ]) -> Any:
        url = f"https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1_1/{category}.jsonl"
        self.dataset = pd.read_json(url, lines=True)
    
    def sample(self):
        sample = self.dataset.sample()
        return sample.to_dict(orient='records')[0]
    

class DROP(Dataset):
    name = "ucinlp/drop"
    paper = "https://arxiv.org/abs/1903.00161v2"
    dataset = None
    categories = None
    def load(self) -> None:
        self.dataset = load_dataset(self.name, trust_remote_code=True)["validation"]

class WinoGrande(Dataset):
    name = "allenai/winogrande"
    paper = "https://arxiv.org/abs/1907.10641"
    dataset = None
    categories =  ['winogrande_xs', 'winogrande_s', 'winogrande_m', 'winogrande_l', 'winogrande_xl', 'winogrande_debiased']

    def load(self, category: Literal['winogrande_xs', 'winogrande_s', 'winogrande_m', 'winogrande_l', 'winogrande_xl', 'winogrande_debiased']) -> None:
        self.dataset = load_dataset(self.name, category, trust_remote_code=True)["validation"]

class Hellaswag(Dataset):
    name = "Rowan/hellaswag"
    paper = "https://arxiv.org/abs/1905.07830"
    dataset = None
    categories = None
    def load(self) -> None:
        self.dataset = load_dataset(self.name, trust_remote_code=True)["validation"]

class PIQA(Dataset):
    name = "ybisk/piqa"
    paper = "https://arxiv.org/abs/1911.11641"
    dataset = None
    categories = None
    def load(self) -> None:
        self.dataset = load_dataset(self.name, trust_remote_code=True)["validation"]

class SIQA(Dataset):
    name = "lighteval/siqa"
    paper = "https://arxiv.org/abs/1904.09728"
    dataset = None
    categories = None
    def load(self) -> None:
        self.dataset = load_dataset(self.name, trust_remote_code=True)["validation"]


class GLUE(Dataset):
    name = "nyu-mll/glue"
    paper = "https://arxiv.org/abs/1804.07461"
    dataset = None
    categories = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    def load(self, category:Literal['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']) -> None:
        try:
            self.dataset = load_dataset(self.name, category, split="train", trust_remote_code=True)
        except Exception as e:
            print(e)
            self.dataset = load_dataset(self.name, category, split="test", trust_remote_code=True)
        else:
            print(f"Failed to load {self.name}/{category}")

class Boolq(Dataset):
    name = "google/boolq"
    paper = "https://arxiv.org/abs/1905.10044v1"
    dataset = None
    categories = None
    def load(self) -> None:
        self.dataset = load_dataset(self.name, trust_remote_code=True)["validation"]