from torchvision import datasets

def download_eurosat(root="data"):
    dataset = datasets.EuroSAT(root=root, download=True)
    return dataset


if __name__ == "__main__":
    ds = download_eurosat()
    print("Number of images:", len(ds))
    print("Classes:", ds.classes)


