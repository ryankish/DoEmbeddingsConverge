from tqdm import tqdm


def compare_input_ids_between_runs(input_ids_run1_path, input_ids_run2_path):
    with open(input_ids_run1_path, "r") as f1, open(input_ids_run2_path, "r") as f2:
        total_lines = sum(1 for _ in f1)
        f1.seek(0)

        with tqdm(total=total_lines, desc="Comparing input IDs") as pbar:
            for i, (line1, line2) in enumerate(zip(f1, f2), start=1):
                if line1 != line2:
                    print(f"\nMismatch at line {i}:")
                    print(f"Run 1: {line1.strip()}")
                    print(f"Run 2: {line2.strip()}")
                    return
                pbar.update(1)

        # Check if one file has more lines than the other
        if f1.readline() or f2.readline():
            print("\nError: Files have different number of lines.")
            return

    print("\nAll input IDs are identical between the two runs.")


def main():
    input_ids_run1_path = "experiments/0/models/1/data/input_ids.txt"

    input_ids_run2_path = "experiments/0/models/2/data/input_ids.txt"

    compare_input_ids_between_runs(input_ids_run1_path, input_ids_run2_path)


if __name__ == "__main__":
    main()
