from .data_processing import ensure_processed_dataset
from .eda import save_example_visualizations


def main() -> None:
    df = ensure_processed_dataset(force_rebuild=True)
    save_example_visualizations(df)
    print("Processed dataset saved to data/retail_sales.csv")
    print("Example visualizations saved to outputs/figures/")


if __name__ == "__main__":
    main()

