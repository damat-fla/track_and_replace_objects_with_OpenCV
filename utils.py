import os

# Helper function to find a file with a specific prefix in a folder
def find_file_with_prefix(folder, prefix):
    # Search for a file that starts with the given prefix (ignoring the extension)
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if name == prefix:
            return os.path.join(folder, filename)
    raise FileNotFoundError(f"No file starting with '{prefix}' found in {folder}")