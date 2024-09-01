def compare_input_ids_between_runs(input_ids_run1_path, input_ids_run2_path):
    with open(input_ids_run1_path, "r") as f1, open(input_ids_run2_path, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print("Error: Files have different number of lines.")
        return

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            print(f"Mismatch at line {i+1}:")
            print(f"Run 1: {line1.strip()}")
            print(f"Run 2: {line2.strip()}")
            return

    print("All input IDs are identical between the two runs.")


def main():
    input_ids_run1_path = "experiments/0/models/1/data/input_ids.txt"

    input_ids_run2_path = "experiments/0/models/2/data/input_ids.txt"

    compare_input_ids_between_runs(input_ids_run1_path, input_ids_run2_path)


if __name__ == "__main__":
    main()
