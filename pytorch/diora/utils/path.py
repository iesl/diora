import os


def package_path():
    my_directory = os.path.dirname(os.path.abspath(__file__))
    my_package_directory = os.path.join(my_directory, '..', '..')
    return os.path.abspath(my_package_directory)
