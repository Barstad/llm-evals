from typing import Optional
import streamlit as st
from data import BBH, GPQA, MATH, Dataset, IFEval, MMLUPro, MuSR


datasets = {
    "MMLU-Pro": MMLUPro,
    "GPQA": GPQA,
    "MuSR": MuSR,
    "MATH": MATH,
    "IFEval": IFEval,
    "BBH": BBH,
}


@st.cache_resource
def load_data(dataset_name: str, category: Optional[str] = None):
    data = datasets[dataset_name]()
    if data.categories is not None:
        data.load(category)
        return data
    data.load()
    return data


def show_example(dataset_name: str, data: Dataset):
    st.subheader(f"Dataset: {dataset_name}")
    st.write(f"Paper: {data.paper}")
    sample = data.sample()
    for key, value in sample.items():
        if key not in ["id", "split"]:
            st.subheader(key.capitalize())
            if dataset_name == "MATH":
                st.markdown(value)
            else:
                st.write(value)


def main():
    st.title("LLM Eval Benchmark datasets")

    dataset_name = st.selectbox("Select a dataset", list(datasets.keys()))
    category = None
    if datasets[dataset_name].categories is not None:
        category = st.selectbox("Select a category", datasets[dataset_name].categories)
    if dataset_name == "GPQA":
        st.write(
            "The authors request we do not display examples from this dataset. Referring to the paper: https://arxiv.org/pdf/2311.12022"
        )
    else:
        if st.button("Show Random Example"):
            print(f"Loading {dataset_name} in category {category}.")
            data = load_data(dataset_name, category)
            show_example(dataset_name, data)


if __name__ == "__main__":
    main()
