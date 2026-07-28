from io.dataset import Dataset


def main():

    dataset = Dataset("bottle")

    dataset.load_images()

    print(f"Loaded {len(dataset)} observations.")


if __name__ == "__main__":
    main()
