import os



def data_root():
    """
    default location of a top-level data directory ('~/data'),
    or read from DATA environment variable.
    """
    if not os.environ.get("DATA"):
        print("Environment variable DATA not set")
    else:
        return os.environ["DATA"]

    root = input("Enter data root: ")
    if not os.path.isdir(root):
        raise ValueError(f"root {root} is not a directory")
    os.environ["DATA"] = root
    return root


def data_dir(dataset_name: str):
    """
    returns the path of dataset_name in the data root directory.
    """
    if dataset_name == "cifar10":
        _path = os.path.join(data_root(),"cifar10") 
        if not os.path.exists(_path):
            os.mkdir(_path)
            print("Created directory: ", _path)
        return _path
            # raise ValueError(f"root of dataset {dataset_name} not found")
    else :
        raise ValueError(f"dataset {dataset_name} not defined")
    
    return 